import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

from UNet2 import UNet, conv_loss
from samples import get_samples, show_samples, RMSELoss, saveRMS, get_training_data
from get_solution import get_solution

if torch.cuda.is_available():
    print("CUDA running")
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def start_training(start_epoch, end_epoch, domain_size, batch_size, unet, dtype=torch.cuda.FloatTensor):
    # # Start training
    T = torch.zeros(batch_size,1,domain_size,domain_size)
    # Initialize a U-net
    # unet = UNet(dtype, img_size=domain_size).type(dtype)
    # create a new closure of conv_loss object
    get_loss = conv_loss(D = 5, dtype = dtype, domain_size = domain_size)
    # Specify optimizer
    optimizer = optim.Adam(unet.parameters(), lr = 2e-4)
    # Mark experiment
    dirName = "1123hollow"
    iteration_number, epochs = 200, end_epoch
    # Getting samples and corresponding solutions
    hollowgeometry = torch.ones(batch_size, 1, domain_size, domain_size)
    center = 10
    d = 3
    hollowgeometry[0,0,center-d:center+d, center-d:center + d] = 0
    sample, solution = get_samples(domain_size, 1, 0.5, 1, 0, hollowgeometry)
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
            T = get_training_data(batch_size, domain_size)
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
        print('epoch{}/{}, loss = {:.3f}'.format(epoch, epochs, epoch_loss))
        # if (epoch == 1) or (epoch % 20 == 0):
        show_samples(solution, unet(sample.type(dtype)), epoch, plotdir, hollowgeometry)
        # RMSE error
        sol = torch.from_numpy(solution)
        error = RMSELoss(unet(sample.type(dtype)), sol, dtype)
        err_list.append(error)
        if epoch % 200 == 0:
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
    parser.add_argument('--start_epoch', type = int, default = 1, help='starting epoch denotation')
    parser.add_argument('--epochs', type=int, default=512, help='number of epochs')
    parser.add_argument('--iscontinue', type=bool, default=False, help='whether or not to continue training from a pretrained model')

    opt = parser.parse_args()
    print(opt)

    if opt.cuda and torch.cuda.is_available():
        print("CUDA running")
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("CUDA available, could have run with --cuda")
        dtype = torch.FloatTensor
    # filename = "0821/LaplaceHist_1024.pth"
    unet = UNet(dtype, img_size=opt.domain_size).type(dtype)
    print(unet)
    if opt.iscontinue:
        unet.load_state_dict(torch.load(filename))
        unet.eval()
    start_training(opt.start_epoch, opt.epochs, opt.domain_size, opt.batch_size, unet, dtype=dtype)
