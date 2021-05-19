import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_obj_loc(singlegeo):
    "return geometry masks of the internal objects"
    # Locate the hollow region
    newgeo = (~singlegeo.astype(bool)).astype(int)
    georows, geocols = np.nonzero(newgeo)
    # Detect the real obstacle boundaries
    centerLeft = geocols[0]
    centerRight = geocols[-1]
    centerUpper = georows[0]
    centerLower = georows[-1]
    return [centerUpper, centerLower, centerLeft, centerRight]

def get_geo_mask(geo):
    "return geometry masks of the internal objects"

    newgeo = (~geo.astype(bool)).astype(int)
    n_sample, georows, geocols = np.nonzero(newgeo)

    georows = georows.reshape(geo.shape[0],16)
    geocols = geocols.reshape(geo.shape[0],16)
    n_sample = n_sample.reshape(geo.shape[0],16)

    pos = np.zeros([geo.shape[0],3,16])
    pos[:,0,:] = n_sample
    pos[:,1,:] = georows
    pos[:,2,:] = geocols
    
    # Inner geometry
    innerpos = pos[:,:, (5,6,9,10)].copy()
    inner_positions = innerpos.transpose((2, 0, 1)).reshape(geo.shape[0]*4,3).astype(int)
    inner_positions = tuple(map(tuple, inner_positions))
    a, b, c = zip(*inner_positions)
    inner_geo = np.ones((geo.shape[0], 1, 32, 32))
    inner_geo[a, 0, b, c] = 0
    
    # object velocity bd
    obj_bd = pos[:,:, (0,1,2,3,4,7,8,11,12,13,14,15)].copy()
    bd_positions = obj_bd.transpose((2, 0, 1)).reshape(geo.shape[0]*12,3).astype(int)
    bd_positions = tuple(map(tuple, bd_positions))
    a, b, c = zip(*bd_positions)
    bd_geo = np.zeros((geo.shape[0], 1, 32, 32))
    bd_geo[a, 0, b, c] = 1



    # object boundary (except corner)
    pressure_inner = pos[:,:, (1,2,4,7,8,11,13,14)].copy()
    pressure_inner_up = pos[:,:, (1,2)]
    pressure_inner_down = pos[:,:, (13,14)]
    pressure_inner_left = pos[:,:, (4,8)]
    pressure_inner_right = pos[:,:, (7,11)]
    pressure_inner_up[:,1,:] -=  1
    pressure_inner_down[:,1,:] +=  1
    pressure_inner_left[:,2,:] -=  1
    pressure_inner_right[:,2,:] +=  1
    pressure_inner_compare = np.concatenate((pressure_inner_up, pressure_inner_down,pressure_inner_left,pressure_inner_right), axis=2)

    pressure_bd_positions = pressure_inner.transpose((2, 0, 1)).reshape(geo.shape[0]*8,3).astype(int)
    pressure_bd_positions = tuple(map(tuple, pressure_bd_positions))
    a, b, c = zip(*pressure_bd_positions)
    pressure_bd_geo = np.zeros((geo.shape[0], 1, 32, 32))
    pressure_bd_geo[a, 0, b, c] = 1


    # object boundary reference (except corner)
    compare_positions = pressure_inner_compare.transpose((2, 0, 1)).reshape(geo.shape[0]*8,3).astype(int)
    compare_positions = tuple(map(tuple, compare_positions))
    a, b, c = zip(*compare_positions)
    compare_geo = np.zeros((geo.shape[0], 1, 32, 32))
    compare_geo[a, 0, b, c] = 1


    # geofocus
    # focus_region = 4
    # focus_area = obj_bd.copy()
    # focus_up = pos[:,:, (0,1,2,3)].copy()
    # focus_down = pos[:,:, (13,14)].copy()
    # focus_left = pos[:,:, (4,8)].copy()
    # focus_right = pos[:,:, (7,11)].copy()
    # geofocus[:,:, max(1, centerUpper-focus_region):min(centerLower+focus_region, domain_size - 1), max(centerLeft-focus_region, 1):min(centerRight+focus_region, domain_size - 1)] = 100
    # geofocus = (geofocus * geometry).detach()

    return inner_geo, bd_geo, pressure_bd_geo, compare_geo