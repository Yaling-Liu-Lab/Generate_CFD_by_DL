import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def conv_loss(domain_size=32):
    "convolutional loss function based on steady heat equation"
    "Neumann condition version"
    h = L / (domain_size-3)
    kernel = torch.tensor([[[[0, 1/4, 0], [1/4, -1, 1/4], [0, 1/4, 0]]]]).type(dtype)
    bd_kernel = torch.tensor([[[[0, 1/4, 0], [-1/2*h, -1, 1/2], [0, 1/4, 0]]]]).type(dtype)
    
    full_size = domain_size 
    img_size = full_size
    reductions = []
    lambd = 128.0
    while img_size > 32:
        img_size /= 4
        indices = np.round(np.linspace(1, full_size-2, img_size)).astype(np.int32)
        indices = np.insert(indices, 0, 0)
        indices = np.append(indices, full_size - 1)
        reductions.append(np.ix_(indices, indices))
    def loss(input_img, isNeum):
        img = input_img[:,:,1:-1,1:-1]
        total_loss = F.conv2d(img, kernel).abs().mean() # main loss from original img
        if isNeum[0]:
            bd = input_img[:,:,1:-1,0:3] # the left boundary region that affect the boundary values on left
            total_loss += lambd * F.conv2d(bd, bd_kernel).abs().mean() # total loss = internel loss+bonudary loss
            for rows, cols in reductions:
                reduced_img = input_img[:,:,rows,cols] # include NMBC
                bd = reduced_img[:,:,1:-1,0:3]
#                 total_loss += F.conv2d(bd, bd_kernel).abs().mean()
        for rows, cols in reductions:
            reduced_img = input_img[:,:,rows,cols]
            total_loss += F.conv2d(reduced_img[:,:,1:-1,1:-1], kernel).abs().mean()
        return total_loss
    return loss

class UNet(nn.Module):
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
                self.encoding_layers.append(nn.Conv2d(1, filters, kernel_size=4, stride=2, padding=1))
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
                self.decoding_layers.append(nn.ConvTranspose2d(filters*2, 1, kernel_size=4, stride=2, padding=1))
            else:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*filters,8*filters)*2, min(2**(i-1)*filters, 8*filters), kernel_size=4, stride=2, padding=1))
            self.decoding_BN.append(nn.BatchNorm2d(min(max(2**(i-1),1)*filters, 8*filters)))

        self.bd = torch.zeros(1,1,img_size,img_size)
        self.bd[:,:,:,0] = 1
        self.bd[:,:,0,:] = 1
        self.bd[:,:,:,-1] = 1
        self.bd[:,:,-1,:] = 1
    
        self.bd = self.bd.type(dtype)

    def forward(self, x, isNeum=[False, False, False, False]):
        ini_state = x
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
        nmbc = 0 * x
        nmbc[:,:,:,0:2] = 1
        nmbc[:,:,0:2,:] = 1
        nmbc[:,:,:,-2:] = 1
        nmbc[:,:,-2:,:] = 1
        if isNeum[0]:
            nmbc[:,:,2:-2,1] = 0
            
        T_results = (x + 1) * 1/2  # x belongs to [-1,1], then (x + 1)* 1/2 *100scaled to 0 ~ 100
        fixed_values = nmbc
        return T_results * (1 - fixed_values) + ini_state * fixed_values
#         return T_results * (1 - self.bd) + ini_state * self.bd