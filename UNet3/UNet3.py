import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"Example of solving (U,V,P)"

def conv_loss(domain_size=32, dtype=torch.FloatTensor):
    "convolutional loss function based on steady heat equation"
    K = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]]).type(dtype) # Must be this one!
    full_size = domain_size
    img_size = full_size
    reductions = []
    inner_reductions = []

    RE = 20
    L = 1
    h = L / (domain_size - 1)
    CFL = 0.04
    dt = CFL * h
    while img_size > 32:
        img_size /= 4
        indices = np.round(np.linspace(0, full_size-1, img_size)).astype(np.int32)
        reductions.append(np.ix_(indices, indices))
        mid /= 4
        D /= 4
        inner_indices = np.round(np.linspace(mid-D, mid+D-1, D)).astype(np.int32)
        inner_reductions.append(np.ix_(indices, indices))

    def loss(img):
        
        "img: channel 0: variable U"
        "img: channel 1: variable V"
        "img: channel 2: variable P"
        
        U = img[:,0:1,:,:]
        V = img[:,1:2,:,:]
        P = img[:,2:3,:,:]
        X_visc = 1/h**2 * 1/RE * F.conv2d(U, K)
        Y_visc = 1/h**2 * 1/RE * F.conv2d(V, K)
        P_visc = F.conv2d(P, K)

        X_adv = U[:,:,1:-1, 1:-1] /(h) * (U[:,:,1:-1, 1:-1] - U[:,:,1:-1, 0:-2])
        X_adv += V[:,:,1:-1, 1:-1] /(h) * (U[:,:,1:-1, 1:-1] - U[:,:,0:-2, 1:-1])
        X_adv += 1/(2*h) * (P[:,:,1:-1, 2:] - P[:,:,1:-1, 0:-2])

        Y_adv = U[:,:,1:-1, 1:-1] /(h) * (V[:,:,1:-1, 1:-1] - V[:,:,1:-1, 0:-2])
        Y_adv += V[:,:,1:-1, 1:-1] /(h) * (V[:,:,1:-1, 1:-1] - V[:,:,0:-2, 1:-1])
        Y_adv += 1 /(2*h) * (P[:,:,2:, 1:-1] - P[:,:,0:-2, 1:-1])

        P_b = ((1/dt/(2*h)*( (U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2])+(V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1]) ) -
            ((U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2]) / (2*h))**2 -
            2*((U[:,:,2:, 1:-1] - U[:,:,0:-2, 1:-1]) / 2/h * (V[:,:,1:-1, 2:] - V[:,:,1:-1, 0:-2]) /2/ h)-
            ((V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1]) / (2*h))**2))
        P_b *= (-1/4*h**2)

        P_neum = (P[:,:,:, -1] - P[:,:,:,-2]).abs().mean()
        P_neum += (P[:,:,0, :] - P[:,:,1, :]).abs().mean() 
        P_neum += (P[:,:,:, 0] - P[:,:,:, 1]).abs().mean() 

        total_loss = (X_visc + X_adv).abs().mean()
        total_loss += (Y_visc + Y_adv).abs().mean()
        total_loss += ( (P_visc + P_b).abs().mean() + P_neum )
        # total_loss = torch.cat((U_adv,RHS_adv), dim=3) + 1/RE/(h**2)*torch.cat((u_visc,rhs_visc), dim=3) - P
        return total_loss
    return loss

# def get_factor(M, m)
#     get_factor = lambda M, m: ((M-m) / 2, -1-2*m/(M-m))
#     U_results = get_factor(UMAX, UMIN)

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
                self.encoding_layers.append(nn.Conv2d(3, filters, kernel_size=4, stride=2, padding=1))
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
                self.decoding_layers.append(nn.ConvTranspose2d(filters*2, 3, kernel_size=4, stride=2, padding=1))
            else:
                self.decoding_layers.append(nn.ConvTranspose2d(min(2**i*filters,8*filters)*2, min(2**(i-1)*filters, 8*filters), kernel_size=4, stride=2, padding=1))
            self.decoding_BN.append(nn.BatchNorm2d(min(max(2**(i-1),1)*filters, 8*filters)))

        self.bd = torch.zeros(1,1,img_size,img_size)
        self.bd[:,:,:,0] = 1
        self.bd[:,:,0,:] = 1
        self.bd[:,:,:,-1] = 1
        self.bd[:,:,-1,:] = 1


        self.p_bd = torch.zeros(1,1,img_size,img_size)
        self.p_bd[:,:,-1,:] = 1
        # mid = img_size // 2
        # d = 5
        # self.bd[:,:,mid -d:mid+d, mid-d:mid+d] = 1
        self.bd = self.bd.type(dtype)
        self.p_bd = self.bd.type(dtype)

    def forward(self, x):

        UMAX, UMIN = 1, -0.1
        VMAX, VMIN = 0.2, -0.2
        PMAX, PMIN = 2.5, -2.5

        get_factor = lambda M, m: ((M-m) / 2, -(M+m)/(M-m))
        U_scale, U_base = get_factor(UMAX, UMIN)
        V_scale, V_base = get_factor(VMAX, VMIN)
        P_scale, P_base = get_factor(PMAX, PMIN)
        ini_states = x.clone()

        ini_state_U = ini_states[:,0:1,:,:]   # channel #0 denotes U
        ini_state_V = ini_states[:,1:2,:,:]   # channel #1 denotes V
        ini_state_P = ini_states[:,2:3,:,:]   # channel #2 denotes p

        # geometry = ini_states[:,1:2,:,:] # channel #1 denotes the geometry
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
        # print(x)
        U_results = (x[:,0:1,:,:] - U_base) * U_scale
        V_results = (x[:,1:2,:,:] - V_base) * V_scale
        P_results = (x[:,2:3,:,:] - P_base) * P_scale

        # T_results = (x[:,0:1,:,:] + 1) * 0.5 # x belongs to [-1,1], then (x + 1)* 50 scaled to 0 ~ 100
        # T_results = T_results * geometry
        # print(T_results.size())

        # Notice for U, V:  boundaries are Dirichelet Boundary
        # For p: Neumann Condition for 3 boundaries, but dirichlet on top
        U_results = U_results * (1 - self.bd) + ini_state_U * self.bd
        V_results = V_results * (1 - self.bd) + ini_state_V * self.bd
        P_results = P_results * (1 - self.p_bd) + ini_state_P * self.p_bd
        
        # print("AFTER BD")
        # print("THIS IS U", U_results)
        # print("THIS IS V", V_results)
        # print("THIS IS P", P_results)

        # 
        # print(geometry.size())
        return torch.cat([U_results, V_results, P_results], dim=1)