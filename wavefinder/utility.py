import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt


def generate_la(GroupPools, L_threshold=1e4, A_threshold=1e4, 
                num_points=2048, base_scale=256, Hs=12):
    '''generate L, A normalized by L_p and H_s from detections. 
    '''
    L = []
    A = []
    num = 0
    for i in GroupPools:
        if len(i) != 1:
            if np.max(np.array(i)[:,2]) > A_threshold:
                num += 1
                continue
            for j in i:
                if j[1] < L_threshold:
                    L.append(j[1] / num_points * base_scale)  # become the true L 
                    A.append(j[2] / Hs)
    print('number of deleted fields: ', num)
    return L, A 

def plot_pdf(L, A, whether_temporal=True):

    grid_points = 2**7  # Grid points in each dimension
    N = 6

    if len(L) > 50000:     
        Data = np.zeros((50000,2))
        Data[:,0] = L[:50000]
        Data[:,1] = A[:50000]
    else:
        Data = np.concatenate((np.atleast_2d(L), np.atleast_2d(A))).T
    
    kde = FFTKDE(bw = 0.12, kernel='gaussian')
    grid, points = kde.fit(Data).evaluate(grid_points)

    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T

    # Plot the kernel density estimate
    fig, ax = plt.subplots()
    ax.contour(x, y, z, N, linewidths=0.8, colors='k')
    cs = ax.contourf(x, y, z, N, cmap="RdBu_r")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_ylabel('$A/H_s$')
    if whether_temporal:
        ax.set_xlabel('$L/T_p$')
    else:
        ax.set_xlabel('$L/L_p$')
    clb = fig.colorbar(cs, ax=ax)
    clb.ax.tick_params(labelsize=12) 
    clb.ax.set_title('$p(L,A)$', fontsize=12)
    
    plt.show()