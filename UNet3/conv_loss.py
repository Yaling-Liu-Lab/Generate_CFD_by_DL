import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_loss(domain_size=32):
    "convolutional loss function based on steady heat equation"
    "internal: indices of the location of internal object"
    kernel = torch.tensor([[[[0, 1/4, 0], [1/4, -1, 1/4], [0, 1/4, 0]]]]).type(dtype)
    full_size = domain_size
    img_size = full_size
    reductions = []
    while img_size > 32:
        img_size /= 4
        indices = np.round(np.linspace(0, full_size-1, img_size)).astype(np.int32)
        reductions.append(np.ix_(indices, indices))
    def loss(img, internal = None):
        total_loss = F.conv2d(img, kernel).abs().mean()
        if internal is not None:

        for rows, cols in reductions:
            total_loss += F.conv2d(img[:,:,rows,cols], kernel).abs().mean()
        return total_loss
    return loss