import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def conv_loss(D = 5, domain_size=32, dtype=torch.FloatTensor):
    "convolutional loss function based on steady heat equation"
    kernel = torch.tensor([[[[0, 1/4, 0], [1/4, -1, 1/4], [0, 1/4, 0]]]]).type(dtype)
    full_size = domain_size
    img_size = full_size
    reductions = []
    mid = domain_size // 2
    # centerLeft = mid - D
    # centerRight = mid + D
    # centerUpper = mid - D
    # centerLower = mid + D
    inner_reductions = []

    while img_size > 32:
        img_size /= 4
        indices = np.round(np.linspace(0, full_size-1, img_size)).astype(np.int32)
        reductions.append(np.ix_(indices, indices))
        mid /= 4
        D /= 4
        inner_indices = np.round(np.linspace(mid-D, mid+D-1, D)).astype(np.int32)
        inner_reductions.append(np.ix_(indices, indices))

    def loss(img):
        "img: channel 0: variable T"
        "img: channel 1: geometry information"
        geometry = img[:,1:2,:,:]
        imgT = img[:,0:1,:,:]

        geo = geometry[0,0,:,:].detach().cpu().numpy()
        newgeo = (~geo.astype(bool)).astype(int)
        georows, geocols = np.nonzero(newgeo)

        # Not a good choice
        centerLeft = geocols[0]
        centerRight = geocols[-1]
        centerUpper = georows[0]
        centerLower = georows[-1]

        total_loss = F.conv2d(imgT, kernel).abs().mean()

        if D > 0:
            total_loss = (total_loss - F.conv2d(imgT[:,:,centerUpper-1:centerLower+1, centerLeft-1:centerRight+1], kernel)).abs().mean()
        # The object of the coarse-grained version remain unsolved for reductions 11/05
        for rows, cols in reductions:
            total_loss += F.conv2d(imgT[:,:,rows,cols], kernel).abs().mean()
        return total_loss
    return loss


class UNet(nn.Module):
    "Physics informed fully conv network autoencoder"
    def __init__(
        self, 
        dtype, 
        img_size = 32, # pre-defined domain_size settings
        filters = 64, # number of filters
    ):
        super().__init__()
        self.image_size = img_size
        self.layers = int(np.log2(img_size)) # number of layers
        self.filters = filters
        self.dtype = dtype

        self.encoding_layers = nn.ModuleList()
        self.encoding_BN = nn.ModuleList()
        for i in range(self.layers):
            if i == 0:
                self.encoding_layers.append(nn.Conv2d(2, filters, kernel_size=4, stride=2, padding=1))
            else:
                self.encoding_layers.append(nn.Conv2d(min(2**(i-1),8)*filters, min(2**i, 8)*filters, kernel_size=4, stride=2, padding=1))
            self.encoding_BN.append(nn.BatchNorm2d(min(2**i*filters, 8*filters)))

        self.encoded = None
    
        self.decoding_layers = nn.ModuleList()
        self.decoding_BN = nn.ModuleList()

        for i in range(self.layers)[::-1]:
            if i == self.layers-1:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*filters, 8*filters), min(2**(i-1)*filters, 8*filters), kernel_size=4, stride=2, padding=1))
            elif i == 0:
                self.decoding_layers.append(nn.ConvTranspose2d(filters*2, 2, kernel_size=4, stride=2, padding=1))
            else:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*filters,8*filters)*2, min(2**(i-1)*filters, 8*filters), kernel_size=4, stride=2, padding=1))
            self.decoding_BN.append(nn.BatchNorm2d(min(max(2**(i-1),1)*filters, 8*filters)))

        self.bd = torch.zeros(1,1,img_size,img_size)
        self.bd[:,:,:,0] = 1
        self.bd[:,:,0,:] = 1
        self.bd[:,:,:,-1] = 1
        self.bd[:,:,-1,:] = 1
        # mid = img_size // 2
        # d = 5
        # self.bd[:,:,mid -d:mid+d, mid-d:mid+d] = 1
        self.bd = self.bd.type(dtype)

    def forward(self, x):
        ini_states = x.clone()

        ini_state = ini_states[:,0:1,:,:] # channel #0 denotes the input variables, T
        geometry = ini_states[:,1:2,:,:] # channel #1 denotes the geometry
        # print(geometry.size())
        x_copy = []
        for i in range(self.layers):
            if i == 0:
                x = F.leaky_relu(self.encoding_layers[i](x), 0.2)
            elif i == self.layers - 1:
                x = self.encoding_layers[i](x)
            else:
                x = F.leaky_relu(self.encoding_BN[i](self.encoding_layers[i](x)), 0.2)
            x_copy.append(x)
        self.encoded = x_copy.pop(-1)
        
        for i in range(self.layers):
            if i == 0:
                x = self.decoding_BN[i](self.decoding_layers[i](F.relu(x)))
            elif i == self.layers - 1:
                x = torch.tanh(self.decoding_layers[i](F.relu(torch.cat((x,x_copy[0]), dim=1))))
            else:
                x = self.decoding_BN[i](self.decoding_layers[i](F.relu(torch.cat((x,x_copy[-1*i]), dim=1))))
        # print("x size = ", x.size())
        T_results = (x[:,0:1,:,:] + 1) * 0.5 # x belongs to [-1,1], then (x + 1)* 50 scaled to 0 ~ 100
        T_results = T_results * geometry
        # print(T_results.size())
        T_results = T_results * (1 - self.bd) + ini_state * self.bd
        # 
        # print(geometry.size())
        return torch.cat([T_results, geometry], dim=1)