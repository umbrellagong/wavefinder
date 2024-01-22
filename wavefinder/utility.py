import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from wave import LinearRandomWave
from detectionbroad import detect_single


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


def plot_pdf(L, A, whether_temporal=True, xlim=(0,3), ylim=(0,1.2), N=6):

    grid_points = 2**7  # Grid points in each dimension

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
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('$A/H_s$')
    if whether_temporal:
        ax.set_xlabel('$L/T_p$')
    else:
        ax.set_xlabel('$L/L_p$')
    clb = fig.colorbar(cs, ax=ax)
    clb.ax.tick_params(labelsize=12) 
    clb.ax.set_title('$p(L,A)$', fontsize=12)
    
    plt.show()
    
    
def generate_la_broad(GroupPools, Hs=12, Tp=15):
    
    A = []
    L = []
    # [A, L, seed, num in seed]
    for i in range(len(GroupPools)):
        for j in range((len(GroupPools[i]))):
            A.append(GroupPools[i][j][2] / Hs)
            L.append(GroupPools[i][j][3] / Tp)    
    return L, A


def plot_groups_broad(seed, kws_wave_spectrum, threshold=6, zero_gap=2, xlim=[0,1000], 
                detection=None):
    # generate wave field 
    wave = LinearRandomWave(**kws_wave_spectrum)
    wave.prepare_wave(seed)
    t_eval = np.arange(0, wave.period, 0.1) 
    elevation = wave.generate_wave(t_eval)
    
    if detection is None:
        detection = detect_single(seed, kws_wave_spectrum, threshold, zero_gap)
    
    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(t_eval, elevation)
    ax1.plot(t_eval, [0] * len(t_eval), '--', color='black')
    ax1.plot(t_eval, [threshold] * len(t_eval), '--', color='red')
    ax1.plot(t_eval, [-threshold] * len(t_eval),'--', color='red')

    for i in detection:
        if (i[1] + i[3]) < wave.period:
            ax1.axvspan(i[1], i[1] + i[3], alpha=0.2, color='tab:red')
        else:
            ax1.axvspan(i[1], t_eval[-1], 
                        alpha=0.2, color='tab:red')
            ax1.axvspan(t_eval[0], (i[1] + i[3]) % wave.period, 
                        alpha=0.2, color='tab:red')

    ax1.set_xlim(xlim)
    ax1.set_ylim(-2*threshold, 2*threshold)
    

    plt.xlabel('$t$')
    plt.ylabel('elevation $\eta$')
    plt.tight_layout()
    plt.show()