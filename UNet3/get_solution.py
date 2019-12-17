import torch
import numpy as np

def get_solution(input_T, geometry, d = 0, dtype = torch.FloatTensor):
    "FDM method to solve laplace eqn"
    "geometry: boolean np array defines the boundary"
    maxIter = 100000
    delta = 1
    T = input_T[0,0,:,:].numpy()
    T_new = np.copy(T)
    iteration = 0
    size = np.size(T[0,:])
    mid = size // 2
    centerLeft = mid - d
    centerRight = mid + d
    centerUpper = mid - d
    centerLower = mid + d
    while iteration < maxIter:
        # if d == 0:
        T_new[1:-1, 1:-1] = ((T[0:-2, 1:-1] + T[2:, 1:-1]) + (T[1:-1,0:-2] + T[1:-1, 2:]))*0.25
        # else:
        #     T_new[1:centerUpper, 1:-1]=((T[0:centerUpper-1, 1:-1] + T[2:centerUpper+1, 1:-1]) + (T[1:centerUpper, 0:-2] + T[1:centerUpper, 2:]))*0.25
        #     T_new[centerLower+1:-1, 1:-1]=((T[centerLower:-2, 1:-1] + T[centerLower+2:, 1:-1]) + (T[centerLower+1:-1, 0:-2] + T[centerLower+1:-1, 2:]))*0.25
        # print(T_new)
        # print(geometry)
        T_new = T_new * geometry
        # if d != 0:
        #     T_new[mid-d:mid+d, mid-d:mid+d] = 0
#         err = np.max(np.abs(T_new - T))        
        err = (T_new - T).flat
        err = np.sqrt(np.dot(err,err))
        if err <= 1e-9:
            return torch.from_numpy(T_new).unsqueeze(0).unsqueeze(0).type(dtype)
        T = np.copy(T_new)
        iteration += 1
    return torch.from_numpy(T).unsqueeze(0).unsqueeze(0).type(dtype)