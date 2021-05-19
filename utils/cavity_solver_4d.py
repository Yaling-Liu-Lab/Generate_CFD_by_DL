import torch
import numpy as np

def build_up_b(b, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = ((1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

def pressure_poisson(p, dx, dy, b, geometry=None):
    pn = np.empty_like(p)
    pn = p.copy()
    nit = 50
    if geometry is not None:
        newgeo = (~geometry.astype(bool)).astype(int)
        # georows, geocols = np.nonzero(newgeo)
        # # Not a good choice
        # centerLeft = geocols[0]
        # centerRight = geocols[-1]
        # centerUpper = georows[0]
        # centerLower = georows[-1]
        # size = geometry.shape[0]
        # geometry_bd_excluded = np.ones((size, size))
        # geometry_bd_excluded[centerUpper+1:centerLower, centerLeft+1:centerRight] = 0
#     if size < 32:
#         centerUpper = 5
#         centerLower = 7
#         centerLeft = 5
#         centerRight = 7
#         geometry_bd_excluded = np.ones((8, 8))
#         geometry_bd_excluded[centerUpper+1:centerLower, centerLeft+1:centerRight] = 0
        
#         geometry = np.ones((8, 8))
#         geometry[centerUpper:centerLower+1, centerLeft:centerRight+1] = 0
        
            
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
#         p[0,0] = 0.1
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        
        if geometry is not None:
            # p = p * geometry_bd_excluded
            p = p * geometry
        # # Neumann BC at the obj        
        #     p[centerUpper:centerLower+1, centerLeft] = p[centerUpper:centerLower+1,centerLeft - 1].copy()
        #     p[centerUpper:centerLower+1, centerRight] = p[centerUpper:centerLower+1,centerRight + 1].copy()
        #     p[centerUpper, centerLeft : centerRight+1] = p[centerUpper - 1, centerLeft : centerRight+1].copy()
        #     p[centerLower, centerLeft : centerRight+1] = p[centerLower + 1, centerLeft : centerRight+1].copy()
        
        p[-1, 0] = 0


    return p

def solve_flow(nt, nx, ny, u_old, v_old, dt, dx, dy, p_old, u0,v0=0,  geometry=None, RE=20):
    "Solve cavity flow that is not regular with outlet region at the end nodes near bottom"
    
    un = np.empty_like(u_old)
    vn = np.empty_like(v_old)
    b = np.zeros((ny, nx))
    u = u_old.copy()
    v = v_old.copy()
    p = p_old.copy()
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b, geometry = geometry)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / (2*dx) *
                        (un[1:-1, 2:] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / (2*dy) *
                        (un[2:, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 *dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         1 / RE * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / (2*dx) *
                       (vn[1:-1, 2:] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / (2*dy) *
                       (vn[2:, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 *dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        1 / RE * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
    # Set boundary condition
        u[0, :]  = u0 # set velocity on cavity lid equal to 1
        u[:, 0]  = 0
        u[:, -1] = 0        
#         u[-1, 0:start_outlet] = 0
        u[-1, :] = u[-2, :]
#         u[-1, end_outlet:] = 0    
        
        v[0, :]  = v0
        v[:, 0]  = 0
        v[:, -1] = 0        
#         v[-1, 0:start_outlet] = 0
        v[-1, :] = v[-2, :]
#         v[-1, end_outlet:] = 0

        if geometry is not None:
            u = u * geometry
            v = v * geometry
        print(np.abs((un[1:-1, 1:-1] - u[1:-1, 1:-1])).mean())
        
        if n > 100 and np.abs((un[1:-1, 1:-1] - u[1:-1, 1:-1])).mean() <= 1e-9 :
            break
#         print(np.abs((un[1:-1] - u[1:-1])).mean())
#     b = build_up_b(b, dt, u, v, dx, dy)
# #     p = pressure_poisson(p, dx, dy, b, geometry = geometry)
    u_sol = u.copy()
    v_sol = v.copy()
    p_sol = p.copy()
    return u_sol, v_sol, p_sol