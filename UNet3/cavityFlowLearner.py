import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

from UNet3 import UNet, conv_loss
from samples import get_samples, show_samples, RMSELoss, saveRMS, get_training_data
from get_solution import get_solution

if torch.cuda.is_available():
    print("CUDA running")
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def start_training(start_epoch, end_epoch, domain_size, batch_size, unet, myoptimizer, warmStart=False, dtype=torch.cuda.FloatTensor):
    # # Start training
    # S = torch.zeros(batch_size,3,domain_size,domain_size)
    # Initialize a U-net
    # unet = UNet(dtype, img_size=domain_size).type(dtype)

    # create a new closure of conv_loss object
    get_loss = conv_loss(domain_size = domain_size, dtype = dtype)
    # Specify optimizer
    # optimizer = optim.Adam(unet.parameters(), lr = 1e-4)
    optimizer = myoptimizer
    # Mark experiment
    dirName = "1211_warm"
    iteration_number, epochs = 400, end_epoch
    # Getting samples and corresponding solutions
    S_sample, s_solution = get_samples(domain_size, u0=1.0, warm_start=warmStart) # S_sample tuple of 4D torch Tensors with initialization

    # sample, solution = get_samples(domain_size, 1, 0.5, 1, 0, hollowgeometry)
    # Create directory for this exp.
    plotdir = dirName + "/plots"
    try:
        # Create target Directory
        os.mkdir(dirName)
        os.mkdir(plotdir)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
            
    err_list = []
    # start training
    start_time = time.clock()
    for epoch in range(start_epoch, epochs + 1):
        # an epoch starts    
        epoch_loss = 0
        for k in range(iteration_number):
            # an iteration starts
            # for j in range(batch_size):
            T = get_training_data(batch_size, domain_size, warm_start=warmStart)
            # T.requires_grad_(True)
            img = T.requires_grad_(True).type(dtype)
            # img = T.type(dtype)
            output = unet(img)
            loss = get_loss(output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            del loss, T
            del output
        # Conv loss tracking    
        epoch_loss = epoch_loss / iteration_number
        print('epoch{}/{}, loss = {:.6f}'.format(epoch, epochs, epoch_loss))
        # if (epoch == 1) or (epoch % 20 == 0):
        generations = unet(S_sample.type(dtype))
        if epoch % 10 == 0:
            show_samples(s_solution[0], s_solution[1], s_solution[2], generations, epoch, plotdir)
        # RMSE error
        u_sol = s_solution[0]
        U_sol = torch.from_numpy(u_sol)
        error = RMSELoss(generations[0,0,:,:], U_sol, dtype) # ONLY PLOT U
        err_list.append(error)
        if epoch > 1 and epoch % 500 == 0:
            # torch.save(unet.state_dict(), "{}/history_{}.pth".format(dirName, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, '{}/history_{}.pth'.format(dirName, epoch))

            print("Prediction on sample with RMSE = {:.3f}".format(error))
        del error
        del epoch_loss
    elapsed = time.clock() - start_time
    average_time = elapsed / (epochs - start_epoch)
    saveRMS(err_list)
    print("Training ended with {} epochs, running {} seconds per epoch".format(epochs, average_time))
    torch.save(unet.state_dict(), '{}/LaplaceHist_{}.pth'.format(dirName, epoch))

if __name__ == '__main__':
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--domain_size', type=int, default = 32, help='size of computational domain')
    parser.add_argument('--batch_size', type=int, default = 1, help='batch size')
    parser.add_argument('--start_epoch', type = int, default = 0, help='starting epoch denotation')
    parser.add_argument('--epochs', type=int, default=512, help='number of epochs')
    parser.add_argument('--iscontinue', type=bool, default=False, help='whether or not to continue training from a pretrained model')

    parser.add_argument('--isWarmStart', type=bool, default=False, help='whether or not to let training start with several time steps')

    opt = parser.parse_args()
    print(opt)

    if opt.cuda and torch.cuda.is_available():
        print("CUDA running")
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("CUDA available, could have run with --cuda")
        dtype = torch.FloatTensor
    filename = "1203_2/history_3500.pth"
    unet = UNet(dtype, img_size=opt.domain_size).type(dtype)
    print(unet)
    optimizer = optim.Adam(unet.parameters(), lr = 1e-4)
    if opt.iscontinue:

        checkpoint = torch.load(filename)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        unet.eval()

    start_training(opt.start_epoch, opt.epochs, opt.domain_size, opt.batch_size, unet, myoptimizer = optimizer, warmStart=opt.isWarmStart, dtype=dtype)
