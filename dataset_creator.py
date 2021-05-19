import torch
import numpy as np
from torch.utils import data


def dataset_creater(domain_size, velocity_mag = 0.5, early_time_steps=20, num_samples = 2000):
    
    size = domain_size
    num_nodes = 100
    lidVecList = np.linspace(0, velocity_mag, num_nodes)
    

    L = 1 # dimensionless LX / LX
    H = 1 # dimensionless LY / LX
    dx = L / (size - 1)
    dy = H / (size - 1)
    CFL = 0.04
    dt = CFL * min(dx , dy)
    RE = 20
    FlowData = torch.zeros((num_samples,3,size,size))
    
    for k in range(0,num_samples):
        u = np.zeros((size, size))
        v = np.zeros((size, size))
        p = np.zeros((size, size))
        
        U0, V0 = np.random.choice(lidVecList, 2)

#         blank_part = np.random.randint(2,domain_size) # From second node to end
#         u0_vector = np.zeros((1,size))
#         u0_vector[0, 0:blank_part] = u0  
        usol, vsol, psol = solve_flow(early_time_steps, size, size, u, v, dt, dx, dy, p, u0=U0, v0=V0)
        FlowData[k,0:1,:,:] = torch.from_numpy(usol) # Channel 0
        FlowData[k,1:2,:,:] = torch.from_numpy(vsol) # Channel 1
        FlowData[k,2:3,:,:] = torch.from_numpy(psol) # Channel 2
        
    torch.save(FlowData, 'FlowData_UV_0130.pt')
    
    return FlowData

class CavityFlowDataset(data.Dataset):
    """Characterizes the cavity flow dataset for training. """
    
    def __init__(self, root_dir, flowfile):
        'Initialization'
        self.flowdata = torch.load(root_dir + flowfile)

    def __len__(self):
        'Denotes the total number of samples'
        return self.flowdata.size()[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.flowdata[index]
        return X