import numpy as np
from scipy.fft import fft,ifft
from scipy.signal import hilbert
from scipy.optimize import minimize
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

from wave import LinearRandomWave 
pi = np.pi


def detect_from_envelope(envelope, coarse_L_max=100, #coarse
            ckbf_threshold=0.4, ckbf_sd=1.2, ckbf_L_min=3, #check before
            L2_sd=1.5, L2_L_max=60, L2_L_min=3, L2_A_min=0.95, L2_A_max=1.2,# L2
            ckaf_threshold=0.75, ckaf_sd=1.5, ckaf_Lmin=5, ckaf_Amin=1,
            coeff_diff=0.075): #check
    try:
        n = envelope.shape[0]
        coarseposition = _generate_coarse(envelope, coarse_L_max)
        fineposition = _generate_fine(coarseposition, envelope)
        firstchecking = _checking_befopt(fineposition, envelope, 
                                        ckbf_threshold, ckbf_sd, ckbf_L_min)
        L2optimizedresults = _L2optimization(firstchecking, envelope, 
                            L2_sd, L2_L_max, L2_L_min, L2_A_min, L2_A_max)
        L2optimizedresults = _L2optimization(firstchecking, envelope, 
                            L2_sd, L2_L_max, L2_L_min, L2_A_min, L2_A_max)
        finalresults = _checking_aftopt(L2optimizedresults, envelope, 
                            ckaf_threshold, ckaf_sd, ckaf_Lmin, ckaf_Amin)
        finalresults = _DeleteOverlap(finalresults, n, coeff_diff)
        finalresults = _DeleteOverlap(finalresults, n, coeff_diff)
    except Exception:
        print("detection failed")
        finalresults = [1]
    return finalresults

def detect_single(seed=0, whether_from_linear=True,
            Hs=12, Tp=15, gamma=0.02, whether_Gaussian=True,
            base_scale=256, num_mode=1024, 
            whether_temporal=True, num_points=2048,
            data_path=None, direct_path=True, data_t=0, starting_line=4,
            coarse_L_max=100, 
            ckbf_threshold=0.4, ckbf_sd=1.2, ckbf_Lim=3,
            L2_sd=1.5, L2_L_max=60, L2_L_min=3, L2_A_min=0.95, L2_A_max=1.1,
            ckaf_threshold=0.75, ckaf_sd=1.5, ckaf_Lmin=5, ckaf_Amin=1, 
            coeff_diff=0.075,
            whether_plot=False, xrange=None):

    if whether_from_linear:
        wave = LinearRandomWave(Hs, Tp, gamma, whether_Gaussian)
        wave.prepare_wave(seed, base_scale, num_mode, whether_temporal)
        elevation, envelope = wave.generate_wave(num_points=num_points, 
                                                 whether_envelope=True)
    else: 
        assert data_path
        if direct_path:
            data_path = data_path
        else:
            data_path = data_path + 'ran-' + str(seed) + '/ELEV.plt'
        with open(data_path, 'r') as f:
            raw_data = f.readlines()
            elevation = [float(raw_data[i].split()[1]) for i in 
                         range(int(starting_line 
                               + (num_points + 1) * data_t / 10) -1, 
                               int(starting_line + num_points 
                               + (num_points + 1) * data_t / 10)-1)]  
        envelope = np.abs(hilbert(elevation))
    
    detection = detect_from_envelope(envelope, coarse_L_max,
            ckbf_threshold, ckbf_sd, ckbf_Lim,
            L2_sd, L2_L_max, L2_L_min, L2_A_min, L2_A_max,
            ckaf_threshold, ckaf_sd, ckaf_Lmin, ckaf_Amin, coeff_diff)

    if whether_plot and len(detection) != 1:
        plt.figure(figsize=(12,4))
        z = np.linspace(-num_points/2, num_points/2, num_points, 
                        endpoint = False)
        plt.plot(z, elevation, label="signal")
        plt.plot(z, envelope, label="envelope")
        for i, group in enumerate(detection):
            if i==len(detection)-1:
                plt.plot(z, group[2] * _Gaussian(z, group[0], group[1]), 'k', 
                         label='detection')
            else:
                plt.plot(z, group[2] * _Gaussian(z, group[0], group[1]), 'k')
        plt.legend()
        plt.xlim(xrange)
        plt.show()

    return detection

def detect_batch(num_seeds, starting_seed, num_p, 
                 save_address, verbose=False, **kw_detection_single):
    if save_address:
        assert type(save_address)==str

    def wrapper(seed):
        if verbose:
            with open('progress', 'a') as f:
                f.write(str(seed) + '\n')
        return detect_single(seed, **kw_detection_single)

    GroupPools = Parallel(n_jobs=num_p)(delayed(wrapper)(seed)
                    for seed in range(starting_seed, starting_seed + num_seeds))
    data = {}
    data['infor'] = kw_detection_single
    data['detections'] = GroupPools

    if save_address:
        np.save(save_address + str(starting_seed), data)
    return GroupPools


# A periodic function decorator
def period(func):
    def wrapper(x, L, n): 
        x = np.mod(x + n/2, n) - n/2
        return func(x, L, n)
    return wrapper

@period
def _d2x(x, L, n):
    # The 
    return (-np.sqrt(2) * (1 - x**2 / L**2) * np.exp(-x**2 / (2 *L**2)) 
            / (2*np.sqrt(pi)*np.sqrt(L**2))) 

@period
def _der1(x, L, n):
    return (np.sqrt(2) * x * (3 - x**2 / L**2) * np.exp(-x**2 / (2 *L**2)) 
        / (2 * np.sqrt(pi) * L**2 * np.sqrt(L**2)))

@period
def _der2(x, L, n):
    return (np.sqrt(2) * (1/2 - 2 * x**2 / L**2 + x**4 / (2*L**4)) 
            * np.exp(-x**2 / (2*L**2)) / (np.sqrt(pi) * L * np.sqrt(L**2)))

def _generate_coarse(envelope, L_max):
    n = envelope.shape[0]
    z = np.linspace(-n/2, n/2, n, endpoint = False)
    coarseresult = np.array([np.real(ifft(fft(_d2x(z, L, n)) * fft(envelope))) 
                                for L in range(1, L_max + 1)])
    coarseresult = np.tile(coarseresult,2)[ : , int(n/2) : int(n+n/2)] 

    def PeriodCoarseResult(i,j): 
        if j > n-1:
            j = j - n
        elif j < 0:
            j = j + n
        return coarseresult[i,j]
    coarseposition = []; 
    for i in range (2, L_max - 2):
        for j in range(0, n):
            subm = np.array([[round(PeriodCoarseResult(n,m), 8) 
                            for m in range(j-2,j+3)] 
                            for n in range(i-2,i+3)])
            if np.argmin(subm) == 12: # number of all points
                coarseposition.append([-int(n/2) + j, i+1])       
    return coarseposition

def _wavelet(para, envelope, whether_derivative=True):
    x, L = para 
    n = envelope.shape[0]
    z = np.linspace(-n/2, n/2, n, endpoint = False)
    value = np.sum((_d2x(x - z, L, n) * envelope))
    if whether_derivative: 
        deriv = np.zeros_like(para)
        deriv[0] = np.sum(_der1(x - z, L, n) * envelope)
        deriv[1] = np.sum(_der2(x - z, L, n) * envelope)
        return value, deriv
    else:
        return value

def _generate_fine(coarseposition, envelope):
    fineposition = []
    for i in coarseposition:
        # Local optimal solution
        solution = minimize(_wavelet, i, envelope, jac=True, 
                            options={'maxiter': 50}) 
        fineposition.append(solution.x)
    fineposition = np.array(fineposition)
    fineposition = fineposition[np.argsort(fineposition[:,0])]
    fineposition[:,1] = fineposition[:,1] / np.sqrt(2)
    return fineposition

def _Gaussian(x, xc, L):          
    return np.exp(-(x-xc)**2 / (2* L **2))

def _criterion_checking(l, position, envelope, sd):
    '''
    l is the length scale for tested wave group, which is the value need to 
    be optimized.
    '''
    n = envelope.shape[0]
    Extendedenvelope = np.tile(envelope, 3)
    center = position + n/2 + n                     # indice 
    lower = int(position + n/2 + n - sd * l)
    upper = int(position + n/2 + n + sd * l)
    detectedwave = Extendedenvelope[lower : upper + 1]
    A = np.max(detectedwave)
    testedwave = A * _Gaussian(np.arange(lower, upper + 1), center, l)
    cjudge = (1 - np.linalg.norm(detectedwave - testedwave) 
                / np.linalg.norm(testedwave))
    return cjudge, A

def _checking_befopt(fineposition, envelope, threshold, sd, L_min):
    checkingresults = []
    for i in fineposition:
        if i[1] > L_min:
            cjudge, A= _criterion_checking(i[1], i[0], envelope, sd)
            if cjudge > threshold:
                checkingresults.append([i[0], i[1], A]) 
    return checkingresults

def _criterion_opt(para, ld, envelope, sd):
    '''
    para: length scale and amplitude
    '''
    n = envelope.shape[0]
    position = para[0]
    lt = para[1]
    A = para[2]
    Extendedenvelope = np.tile(envelope, 3)
    center = position + n/2 + n                     # indice 
    lower = int(position + n/2 + n - sd * ld)
    upper = int(position + n/2 + n + sd * ld)
    detectedwave = np.array(Extendedenvelope[lower : upper + 1])
    testedwave = A * _Gaussian(np.arange(lower, upper + 1), center, lt)
    cjudge = (1 - sum(abs(detectedwave - testedwave) 
                        / (testedwave/2 + detectedwave/2)) 
                    / detectedwave.shape[0])
    return -cjudge

def _L2optimization(checkingresults, envelope, sd, L_max, L_min, 
                                    A_min_coeff, A_max_coeff):
    L2optimizedresults = []
    for i in checkingresults:
        solution = minimize(_criterion_opt, i[:3], 
                            args = (i[1], envelope, sd), 
                            method='L-BFGS-B', 
                            bounds = [(i[0] - 30, i[0] + 30), 
                            (L_min, np.min([1.3 * i[1], L_max])), 
                            (A_min_coeff * i[2], A_max_coeff * i[2])])
        L2optimizedresults.append(np.append(solution.x, solution.fun))
    return L2optimizedresults

def _checking_aftopt(L2optimizedresults, envelope, threshold, sd, L_min, A_min):
    checkingresults = []
    for i in L2optimizedresults:
        cjudge= _criterion_opt(i[:3], i[1], envelope, sd)
        if abs(cjudge) > threshold and i[1] > L_min and i[2] > A_min:
            checkingresults.append([i[0], i[1], i[2], abs(cjudge)]) 
    return checkingresults

def _DeleteOverlap(finalresults, n, coeff_diff=0.075):
    i = 0
    while i < len(finalresults):
        if i == len(finalresults) -1:
            if  (abs(finalresults[i][0] - (finalresults[0][0] + n)) < 
                1.5 * (max(finalresults[i][1],finalresults[0][1]))):
                if abs(finalresults[i][3] - finalresults[0][3]) > coeff_diff:
                    if finalresults[i][3] > finalresults[0][3]:
                        finalresults.pop(0)
                    else:
                        finalresults.pop(i)
                else:
                    if finalresults[i][1] > finalresults[0][1]: 
                        finalresults.pop(0)
                    else:
                        finalresults.pop(i) 
            break       
        else:
            if (abs(finalresults[i][0] - finalresults[i+1][0]) < 
                1.5 * (max(finalresults[i][1],finalresults[i+1][1]))):
                if abs(finalresults[i][3] - finalresults[i+1][3]) > coeff_diff: 
                    if finalresults[i][3] > finalresults[i+1][3]:
                        finalresults.pop(i+1)
                    else:
                        finalresults.pop(i)
                else:
                    if finalresults[i][1] > finalresults[i+1][1]: 
                        finalresults.pop(i+1)
                    else:
                        finalresults.pop(i) 
            else:
                i = i + 1
    return finalresults


if __name__ == "__main__":
    detect_batch(10000, 0, 60, 'spatial', verbose=True,
              Hs=9, whether_temporal=False, base_scale=128, L2_A_max=1.1, 
              ckaf_threshold=0.72, L2_L_max=60, coeff_diff=0.085, 
              whether_plot=False)