import numpy as np
from scipy.fft import fft,ifft
from scipy.signal import hilbert
from scipy.optimize import minimize
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from wave2d import LinearRandomWave2D
from sympy import *                   

plt.rcParams.update({'font.size': 16})


def detect_from_envelope(envelope, peak_amp_threshold=4):
    # first find peak
    pos, amp, d2x, d2y = find_peak(envelope, peak_amp_threshold)
    # find the optimal parameters w.r.t. minimal L2 error
    # bound in terms of the point 
    bounds = [[2, 30]] * len(pos) * 2 + [[-0.75, 0.75]] * len(pos)
    init_x = np.sqrt(abs((-2) * amp / d2x))
    init_y = np.sqrt(abs((-2) * amp / d2y))
    init = np.concatenate((init_x, init_y, np.zeros(len(pos))))
    #init = np.ones(pos.shape[0] * 2) * 3
    results = minimize(error, init, jac=True, method='L-BFGS-B', 
                       bounds=bounds, 
                       args=(pos, amp, envelope), 
                       options={'maxiter': 300})
    
    return results.x, pos, amp
 
 
def detect_single(seed=0, Hs=12, Tp=15, gamma=0.015, theta=np.pi/8, 
                  whether_Gaussian=True,
                  base_scale=32, num_mode=32*4, 
                  num_points=32*8, num_p=6, t=0, cut=32*2,
                  peak_amp_threshold=4,
                  whether_plot=True):
    '''currently only deal with linear wave field
    '''
    
    wave2d = LinearRandomWave2D(Hs, Tp, gamma, theta, whether_Gaussian)
    wave2d.prepare_wave(seed, base_scale, num_mode)
    mesh, env = wave2d.generate_envelope(num_points, num_p, cut, t_series=[t])
    env = env[0]
    L, pos, amp = detect_from_envelope(env, peak_amp_threshold)
    
    if whether_plot:
        _, elev = wave2d.generate_wave(num_points, num_p, [t])
        elev = elev[0]
        
        re_env, _ = reconstruct(L, pos, amp, env)
        levels = np.arange(-12, 12.1, 2)
        
        plt.figure(figsize=(8,6))
        plt.contourf(mesh[0], mesh[1], elev, levels=levels)
        plt.contour(mesh[0], mesh[1], env, levels=levels)
        plt.scatter(mesh[0][0][pos[:,0]], mesh[1][:,0][pos[:,1]], color='r')
        plt.colorbar()
        plt.title('data')
        plt.show()
    
        plt.figure(figsize=(8,6))
        plt.contourf(mesh[0], mesh[1], elev, levels=levels)
        plt.contour(mesh[0], mesh[1], re_env, levels=levels)
        plt.scatter(mesh[0][0][pos[:,0]], mesh[1][:,0][pos[:,1]], color='r')
        plt.colorbar()
        plt.title('reconstruction')
        plt.show()
        
    return L, pos, amp
    
    

def generate_funcs():
    '''let the sympy compute the analytical derivative.'''
    a, x, xc, y, yc, l_x, l_y, rho = symbols('a x xc y yc l_x l_y rho')
    
    G = a * exp(-((x - xc)**2 / l_x**2 + (y - yc)**2 / l_y**2 - 
                2 * rho * (x-xc) * (y-yc)  / (l_x * l_y)) / ((1 - rho**2)))

    G_value = lambdify([x, y, l_x, l_y, rho, xc, yc, a], G)
    dG_lx = lambdify([x, y, l_x, l_y, rho, xc, yc, a], 
                     diff(G, l_x, 1) / G)
    dG_ly = lambdify([x, y, l_x, l_y, rho, xc, yc, a], 
                     diff(G, l_y, 1) / G)
    dG_rho = lambdify([x, y, l_x, l_y, rho, xc, yc, a], 
                      diff(G, rho, 1) / G)
    return G_value, dG_lx, dG_ly, dG_rho

G_value, dG_lx, dG_ly, dG_rho = generate_funcs()


def Gaussian(x, y, l_x, l_y, rho, pos, a): 
    ''' 
    x, y: the pos
    l_x, l_y: the characteristic length of x and y direction
    rho: the correlation coefficient
    pos: the pos of mode
    a: amplitude
    '''
    xc, yc = pos
    value = G_value(x, y, l_x, l_y, rho, xc, yc, a)
    deriv = (value * dG_lx(x, y, l_x, l_y, rho, xc, yc, a),
             value * dG_ly(x, y, l_x, l_y, rho, xc, yc, a),
             value * dG_rho(x, y, l_x, l_y, rho, xc, yc, a))
    return value, deriv


def period_index(j, n): 
    if j > -1:
        j = j - n
    elif j < 0:
        j = j + n
    return j


def find_peak(envelope, amp_threshold):
    '''get the local peak from its 3-level neighbours.
    '''
    
    size_x, size_y = envelope.shape
    coarseposition = []; 
    amp = []; 
    d2x = [];
    d2y = [];
    # judge every point
    for i in range(size_x):
        for j in range(size_y):
            subm = np.array([[envelope[period_index(n, size_x), 
                              period_index(m, size_y)]
                            for m in range(j - 3, j + 4)] 
                            for n in range(i - 3, i + 4)])
            if np.argmax(subm) == 12 and envelope[i,j] > amp_threshold:
                d2x_c = (envelope[period_index(i-1, size_x), j]
                         + envelope[period_index(i+1, size_x), j] 
                         - 2 * envelope[i,j])
                d2y_c = (envelope[i, period_index(j-1, size_y)] 
                         + envelope[i, period_index(j+1, size_y)] 
                         - 2 * envelope[i, j])
                # get the amp, pos and d2 as approximation of l_x, l_y
                coarseposition.append([j, i]) 
                amp.append(envelope[i, j])
                d2x.append(d2x_c)
                d2y.append(d2y_c)
                
    return np.array(coarseposition), np.array(amp), np.array(d2x), np.array(d2y)
    
    
def error(L, pos, amp, envelope):
    ''' compute the error (and der) between reconstructed wave field and true 
        wave field
    
    L:         Gaussian modes parameters (Lx_list, Ly_list, rho_list)
    pos, amp:  determined mode pos list and amp list in find peak
    envelope:  the true envelope map to be approximated 
    '''
    
    G, G_der = reconstruct(L, pos, amp, envelope)
    # summation of L2 difference of everypoint 
    value = np.sum((G - envelope)**2)
    # summation of derivative of L2 difference of everypoint 
    derivative = np.sum(2 * G_der * (G-envelope), axis=(1,2))
    return value, derivative

def reconstruct(L, pos, amp, envelope):
    ''' compute the reconstructed wave field and its der w.r.t L parameters. 
    '''
    n_x, n_y = envelope.shape
    G = np.zeros((n_x, n_y))  
    # Each mode contribute to the derivative of their own modes' parameters
    G_der = np.zeros((3 * len(pos), n_x, n_y)) 
    
    mesh_x, mesh_y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    L_x, L_y, rho = np.split(L, 3)
    for i in range(len(pos)):
        G_temp, G_der[[i, i + len(pos), 
                       i + 2 * len(pos)]] = Gaussian(mesh_x, mesh_y, 
                                                     L_x[i], L_y[i], 
                                                     rho[i], pos[i], amp[i])
        G = G + G_temp

    return G, G_der
    
