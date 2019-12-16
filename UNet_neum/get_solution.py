import torch
import numpy as np


def get_solution(input_T, isNeum, dtype = torch.FloatTensor):
    "FDM method to solve laplace eqn"
    "a denotes the Neumann boundary condition at X = 0"
    maxIter = 1e8
    padT = input_T[0,0,:,:].numpy()
    output_T = input_T.clone().numpy()
    # READ NEUMANN BC FROM INPUT_T
    if isNeum[0]:
        nbc_left = padT[2:-2,0]
#         print(nbc_left)
    if isNeum[1]:
        nbc_upper = padT[0,2:-2]
    if isNeum[2]:
        nbc_right = padT[2:-2,-1]
    if isNeum[3]:
        nbc_bottom = padT[-1,2:-2]
    
    # Acquire the real compute domain of T   
    T = padT[1:-1,1:-1]
    L = 1
    h = L / np.size(padT[0,:])
    T_new = np.copy(T)
    iteration = 0
    while iteration < maxIter:
        T_new[1:-1, 1:-1] = ((T_new[0:-2, 1:-1] + T_new[2:, 1:-1]) + (T_new[1:-1,0:-2] + T_new[1:-1, 2:]))*0.25
        if isNeum[0]:
            T_new[1:-1,0] = 1/3 * (4*T_new[1:-1,1] - T_new[1:-1, 2]  - 2*h*nbc_left)       
        err = (T_new - T).flat
        err = np.sqrt(np.dot(err,err))
        if err <= 1e-12:
            output_T[0,0,1:-1,1:-1] = T_new
            return torch.from_numpy(output_T).type(dtype)
        T = np.copy(T_new)
        iteration += 1
    output_T[0,0,1:-1,1:-1] = T_new    
    return torch.from_numpy(output_T).type(dtype)
