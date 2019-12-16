import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from get_solution import get_solution
import matplotlib.pyplot as plt

def get_training_data(batch_size, domain_size):
    "Get weakly supervised learning data"
    img = torch.zeros(batch_size,2,domain_size,domain_size)
    T = torch.zeros(batch_size,1,domain_size,domain_size)
    T[:,:,:,0] = np.random.uniform(0,1)
    T[:,:,0,:] = np.random.uniform(0,1)
    T[:,:,:,-1] = np.random.uniform(0,1)
    T[:,:,-1,:] = np.random.uniform(0,1)
    geometry = torch.ones(batch_size,1,domain_size,domain_size)
    d = 4
    center = np.random.randint(d + 2, domain_size - d)
    geometry[:,0,center-d:center+d, center-d:center + d] = 0

    img[:,0,:,:] = T # Channel 0
    img[:,1,:,:] = geometry # Channel 1
    return img

def get_samples(size,top,bottom,left,right,geometry,d=5):   
    T = torch.zeros(1,1,size,size)
    T[:,:,0,:] = top
    T[:,:,-1,:] = bottom
    T[:,:,:,0] = left
    T[:,:,:,-1] = right
    
    T[:,:,0,0] = (top + left) / 2
    T[:,:,0,-1] = (top + right) / 2
    T[:,:,-1,0] = (bottom + left) / 2
    T[:,:,-1,-1] = (bottom + right) / 2
    # mid = size // 2
    # d = 5
    T = T * geometry
    # T[:,:,mid -d:mid+d, mid-d:mid+d] = 0
    geo = geometry[0,0,:,:].cpu().numpy()
    sample = torch.cat([T, geometry], dim = 1)
    solution = get_solution(T, geo).cpu().numpy()[0,0,:,:] # solution in 2D numpy, while T in 4D tensor
    return sample, solution

def show_samples(f_0, n_0, epoch, dirName, geometry, d=1):
    "f_0 solution"
    "n_0 prediction"

    size = f_0.shape[0]
    n_0 = n_0.cpu().detach().numpy()[0,0,:,:]

    if d != 0:
        geo = geometry[0,0,:,:].cpu().numpy()
        geo[geo == 0] = np.nan
        f_0  = f_0 * geo
        n_0  = n_0 * geo
    #     f_0[size//2 - d + 1 : size//2 + d - 1, size//2 -d + 1 : size//2 + d - 1] = np.nan
    #     n_0[0,0, size//2 -d + 1 : size//2 + d - 1, size//2 -d + 1 : size//2 + d - 1] = np.nan

    plt.figure(figsize=(16, 10))
    
    plt.subplot(1,2,1)
    plt.imshow(n_0, vmin=0, vmax=1, cmap=plt.cm.jet)
    plt.axis('equal')

    plt.subplot(1,2,2)
    plt.imshow(f_0, vmin=0, vmax=1, cmap=plt.cm.jet)
    plt.axis('equal')

    plt.savefig('{}/predict_epoch{}.png'.format(dirName, epoch))
    plt.close()

def RMSELoss(yhat, y, dtype=torch.FloatTensor):
    rmse = torch.sqrt(torch.mean((yhat.type(dtype) - y.type(dtype))**2))
    return float(rmse)
    
def saveRMS(err_list):
    plt.figure()
    plt.plot(range(1,len(err_list) + 1), err_list)
    plt.ylabel('RMS error')
    plt.xlabel('epochs')
    plt.savefig('rms.png')
    plt.close()