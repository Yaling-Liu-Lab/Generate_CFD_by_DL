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

    # Starting coordinates and ending coordinates for cases w/ outlets
    start_outlet = 1
    end_outlet = 31
    while img_size > 32:
        img_size /= 4
        indices = np.round(np.linspace(0, full_size-1, img_size)).astype(np.int32)
        reductions.append(np.ix_(indices, indices))
        mid /= 4
        D /= 4
        inner_indices = np.round(np.linspace(mid-D, mid+D-1, D)).astype(np.int32)
        inner_reductions.append(np.ix_(indices, indices))
        
    geometry = torch.ones(1, 1, 32, 32).type(dtype)
    centerLeft = 24
    centerRight = 27
    centerUpper = 24
    centerLower = 27
    geometry[:,:,centerLeft:centerRight + 1, centerUpper:centerLower+1] = 0
    
    geometry_bd_excluded = geometry.clone() # denotes the inner object excluded the boundary of the inner obj
    # Not a good choice
    geometry_bd_excluded[:,:,centerUpper, centerLeft:centerRight+1] = 1
    geometry_bd_excluded[:,:,centerLower, centerLeft:centerRight+1] = 1
    geometry_bd_excluded[:,:,centerUpper:centerLower+1, centerLeft] = 1
    geometry_bd_excluded[:,:,centerUpper:centerLower+1, centerRight] = 1
    def loss(ini, output):
        
        "output: channel 0: variable U"
        "output: channel 1: variable V"
        "output: channel 2: variable P"
        
        U = output[:,0:1,:,:]
        V = output[:,1:2,:,:]
        P = output[:,2:3,:,:]
        
        BC_upper = ini[:,0:2,0:1,:]   
#         BC_lowerWall_L = ini[:, 0:3, -1:, 0:start_outlet]
#         BC_lowerWall_R = ini[:, 0:3, -1:, end_outlet:]
        BC_leftWall = ini[:,0:2,:,0:1]
        BC_rightWall = ini[:,0:2,:,-1:]
        BC_lowerWall = ini[:, 2, -1:, :] # Denotes the pressure Solid Wall conditions!
        bd_MSE = torch.mean( (BC_upper - output[:,0:2,0:1,:])**2 )
        bd_MSE += torch.mean( (BC_lowerWall - output[:, 2, -1:, :])**2 ) #
        bd_MSE += torch.mean( (BC_leftWall - output[:,0:2,:,0:1])**2 )
        bd_MSE += torch.mean( (BC_rightWall - output[:,0:2,:,-1:])**2 )
        
        BC_innner_up =  ini[:,0:2,centerUpper, centerLeft:centerRight+1]
        BC_innner_down =  ini[:,0:2,centerLower, centerLeft:centerRight+1]
        BC_innner_left =  ini[:,0:2,centerUpper:centerLower+1, centerLeft]
        BC_innner_right =  ini[:,0:2,centerUpper:centerLower+1, centerRight]
                
        # NEED TO HAVE +1 HERE BECAUSE THE PYTHON INTERVAL ARRANGING, the centerLeft. etc. is detected not assigned
        bd_MSE_inner = torch.mean( (BC_innner_up - output[:,0:2,centerUpper, centerLeft:centerRight+1])**2 )
        bd_MSE_inner += torch.mean( (BC_innner_down - output[:,0:2,centerLower, centerLeft:centerRight+1])**2  )
        bd_MSE_inner += torch.mean( (BC_innner_left - output[:,0:2,centerUpper:centerLower+1, centerLeft])**2 )
        bd_MSE_inner += torch.mean( (BC_innner_right - output[:,0:2,centerUpper:centerLower+1, centerRight])**2 )

        X_visc = 1/h**2 * 1/RE * F.conv2d(U, K)
        Y_visc = 1/h**2 * 1/RE * F.conv2d(V, K)
        P_visc = 1/4 * F.conv2d(P, K)

        X_adv = U[:,:,1:-1, 1:-1] /(2*h) * (U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2])
        X_adv += V[:,:,1:-1, 1:-1] /(2*h) * (U[:,:,2:, 1:-1] - U[:,:,0:-2, 1:-1])
        X_adv += 1/(2*h) * (P[:,:,1:-1, 2:] - P[:,:,1:-1, 0:-2])

        Y_adv = U[:,:,1:-1, 1:-1] /(2*h) * (V[:,:,1:-1, 2:] - V[:,:,1:-1, 0:-2])
        Y_adv += V[:,:,1:-1, 1:-1] /(2*h) * (V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1])
        Y_adv += 1 /(2*h) * (P[:,:,2:, 1:-1] - P[:,:,0:-2, 1:-1])

        P_b = ((1/dt/(2*h)*( (U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2])+(V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1]) ) -
            ((U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2]) / (2*h))**2 -
            2*((U[:,:,2:, 1:-1] - U[:,:,0:-2, 1:-1]) / 2/h * (V[:,:,1:-1, 2:] - V[:,:,1:-1, 0:-2]) /2/ h)-
            ((V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1]) / (2*h))**2))
        P_b *= (+1/4*h**2)

        P_neum = (P[:,:,:, -1] - P[:,:,:,-2]).abs().mean()
        P_neum += (P[:,:,0, :] - P[:,:,1, :]).abs().mean() 
        P_neum += (P[:,:,:, 0] - P[:,:,:, 1]).abs().mean() 
        P_neum += (P[:,:, centerUpper+1:centerLower, centerLeft] - P[:,:,centerUpper+1:centerLower,centerLeft-1]).abs().mean()
        P_neum += (P[:,:, centerUpper+1:centerLower, centerRight] - P[:,:,centerUpper+1:centerLower,centerRight+1]).abs().mean()
        P_neum += (P[:,:, centerUpper, centerLeft : centerRight] - P[:,:,centerUpper-1, centerLeft : centerRight]).abs().mean()
        P_neum += (P[:,:, centerLower, centerLeft : centerRight] - P[:,:,centerLower+1, centerLeft : centerRight]).abs().mean()
#         Velocity_neum = 10 * ( output[:, 0:2, -1, start_outlet:end_outlet] - output[:, 0:2, -2, start_outlet:end_outlet] ).abs().mean()
        Velocity_neum = 10 * ( output[:, 0:2, -1, 1:-1] - output[:, 0:2, -2, 1:-1] ).abs().mean()
        
        
        # Forgot to add this at 0421
        X_loss = (X_visc + X_adv) * geometry[:,:,1:-1, 1:-1]
        Y_loss = 1 * (Y_visc + Y_adv) * geometry[:,:,1:-1, 1:-1]
        P_loss = 1 * (P_visc + P_b)  * geometry[:,:,1:-1, 1:-1]
        
        total_loss = X_loss.abs().mean()
        total_loss += Y_loss.abs().mean()
        total_loss += P_loss.abs().mean() + P_neum + Velocity_neum
#         print('the internal residue loss is:', float(total_loss))
        total_loss += 10 * bd_MSE
        total_loss += (10 * bd_MSE_inner)
        return total_loss, float(bd_MSE), float(bd_MSE_inner), float(P_neum), float(Velocity_neum)
    return loss




def stokes_loss(domain_size=32, dtype=torch.FloatTensor):
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
    Factor = torch.ones(1,1,30,30).type(dtype) 
    Factor[0,0,0:3,:] = 1
    while img_size > 32:
        img_size /= 4
        indices = np.round(np.linspace(0, full_size-1, img_size)).astype(np.int32)
        reductions.append(np.ix_(indices, indices))
        mid /= 4
        D /= 4
        inner_indices = np.round(np.linspace(mid-D, mid+D-1, D)).astype(np.int32)
        inner_reductions.append(np.ix_(indices, indices))

    def loss(ini, output):
        
        "output: channel 0: variable U"
        "output: channel 1: variable V"
        "output: channel 2: variable P"
        
        U = output[:,0:1,:,:]
        V = output[:,1:2,:,:]
        P = output[:,2:3,:,:]
        
        BC_upper = ini[:,0:3,0:1,:]   
        BC_lowerWall = ini[:,0:3,-1:,:] 
        BC_leftWall = ini[:,0:3,:,0:1]
        BC_rightWall = ini[:,0:3,:,-1:]
        
        bd_MSE = torch.mean( (BC_upper - output[:,0:3,0:1,:])**2 )
        bd_MSE += torch.mean( (BC_lowerWall - output[:,0:3,-1:,:])**2 )
        bd_MSE += torch.mean( (BC_leftWall - output[:,0:3,:,0:1])**2 )
        bd_MSE += torch.mean( (BC_rightWall - output[:,0:3,:,-1:])**2 )
                
        
#         print("bd is ", bd_MSE)
        X_visc = 1/h**2 * 1/RE * F.conv2d(U, K)
        Y_visc = 1/h**2 * 1/RE * F.conv2d(V, K)
        P_visc = 1/4 * F.conv2d(P, K)

#         X_adv = U[:,:,1:-1, 1:-1] /(2*h) * (U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2])
#         X_adv += V[:,:,1:-1, 1:-1] /(2*h) * (U[:,:,2:, 1:-1] - U[:,:,0:-2, 1:-1])
#         X_adv += 1/(2*h) * (P[:,:,1:-1, 2:] - P[:,:,1:-1, 0:-2])

#         Y_adv = U[:,:,1:-1, 1:-1] /(2*h) * (V[:,:,1:-1, 2:] - V[:,:,1:-1, 0:-2])
#         Y_adv += V[:,:,1:-1, 1:-1] /(2*h) * (V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1])
#         Y_adv += 1 /(2*h) * (P[:,:,2:, 1:-1] - P[:,:,0:-2, 1:-1])

#         P_b = ((1/dt/(2*h)*( (U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2])+(V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1]) ) -
#             ((U[:,:,1:-1, 2:] - U[:,:,1:-1, 0:-2]) / (2*h))**2 -
#             2*((U[:,:,2:, 1:-1] - U[:,:,0:-2, 1:-1]) / 2/h * (V[:,:,1:-1, 2:] - V[:,:,1:-1, 0:-2]) /2/ h)-
#             ((V[:,:,2:, 1:-1] - V[:,:,0:-2, 1:-1]) / (2*h))**2))
#         P_b *= (+1/4*h**2)

#         P_neum = (P[:,:,:, -1] - P[:,:,:,-2]).abs().mean()
#         P_neum += (P[:,:,0, :] - P[:,:,1, :]).abs().mean() 
#         P_neum += (P[:,:,:, 0] - P[:,:,:, 1]).abs().mean() 

        X = X_visc

        total_loss = X.abs().mean()
        total_loss += (1 * (Y_visc).abs().mean())
        total_loss += (1 * (P_visc).abs().mean())
#         print(total_loss)
        total_loss += 10 * bd_MSE
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
#         self.bd[:,:,:,0] = 1
#         self.bd[:,:,0,:] = 1
#         self.bd[:,:,:,-1] = 1
#         self.bd[:,:,-1,:] = 1

        self.p_bd = torch.zeros(1,1,img_size,img_size)
#         self.p_bd[:,:,-1,:] = 1
        # mid = img_size // 2
        # d = 5
        # self.bd[:,:,mid -d:mid+d, mid-d:mid+d] = 1
        self.bd = self.bd.type(dtype)
        self.p_bd = self.bd.type(dtype)

    def forward(self, x):

#         UMAX, UMIN = 1.0, -1.0
#         VMAX, VMIN = 2.0, -1.0
#         PMAX, PMIN = 8.5, -2.5
        UMAX, UMIN = 0.5, -0.25
        VMAX, VMIN = 1.0, 0
        PMAX, PMIN = 3, -0.25
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