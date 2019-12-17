import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .get_solution import get_solution
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

def show_samples(solution, prediction4D, epoch, dirName, geometry, d=1):
    "f_0 solution"
    "n_0 prediction"

    size = solution.shape[0]
    prediction = prediction4D.cpu().detach().numpy()[0,0,:,:]

    if d != 0:
        geo = geometry[0,0,:,:].cpu().numpy()
        geo[geo == 0] = np.nan
        solution  = solution * geo
        prediction  = prediction * geo
    #     f_0[size//2 - d + 1 : size//2 + d - 1, size//2 -d + 1 : size//2 + d - 1] = np.nan
    #     n_0[0,0, size//2 -d + 1 : size//2 + d - 1, size//2 -d + 1 : size//2 + d - 1] = np.nan

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    L1_distance = np.abs(prediction - solution)

    ax1.imshow(prediction, vmin=0, vmax=1, cmap=plt.cm.inferno)
    ax1.set_title("Generation")

    ax2.imshow(solution, vmin=0, vmax=1, cmap=plt.cm.inferno)
    ax2.set_title("FDM Solution")

    IM = ax3.imshow(L1_distance, vmin=0, vmax=1, cmap=plt.cm.inferno)
    ax3.set_title("L1 distance")

    cb_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
    cbar = fig.colorbar(IM, cax=cb_ax, orientation = "horizontal")

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