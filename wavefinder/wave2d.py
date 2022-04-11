import numpy as np
from scipy.signal import hilbert
import scipy.integrate as integrate
from joblib import Parallel, delayed
from wave import LinearRandomWave


pi = np.pi
g = 9.81

class LinearRandomWave2D(LinearRandomWave):
    '''Generate spatial wave from a frequence spectrum. The generation of 
       temporal wave is trivial from 1D case. 
    '''
    def __init__(self, Hs=12, Tp=15, gamma=0.02, theta=pi/8, 
                 whether_Gaussian=True):
        super().__init__(Hs, Tp, gamma, whether_Gaussian)
        self.theta = theta

    def prepare_wave(self, seed, base_scale=32, num_mode=32*4):
        # specify spectrum form 
        if self.whether_Gaussian:
            alpha = (self.Hs / 4)**2 / (self.gamma * np.sqrt(2 * pi))
            S = self._spectrum_gaussian
        else:
            integration = integrate.quad(self._spectrum_jonswap_single, 
                                         0, 100 * self.wp, 
                                         args=(1, self.wp, self.gamma))[0]
            alpha = (self.Hs / 4) **2 / integration
            S = self._spectrum_jonswap 

        # generate random phase for each (kx, ky) mode
        # num_mode in x direction and 2*num_mode + 1 in y direction 
        np.random.seed(seed)
        self.random_phase = np.random.rand(num_mode, 2 * num_mode + 1) * 2*pi
        
        # generate the amplitude for each mode 
        base = self.kp / base_scale
        self.Amplitude = np.zeros_like(self.random_phase)
        ky_list = np.arange(-num_mode, num_mode + 1) * base
        for i, kx in enumerate(np.arange(1, num_mode+1) * base):
            angle_list = np.arctan(ky_list / kx)
            k_list = np.sqrt(kx**2 + ky_list**2)
            w_list = np.sqrt(g * k_list)
            self.Amplitude[i] = np.sqrt(g**2 / w_list**3
                                   * S(w_list, alpha, self.wp, self.gamma)
                                   * self._spreading(angle_list, self.theta)
                                   * base**2)

        self.period = self.Lp * base_scale
        self.num_mode = num_mode
        self.base = base
        self.base_scale = base_scale
    
    def generate_wave(self, num_points=32*8, num_p=6, t_series=[0]):
        '''parallely generate the wave field for t series.
        '''
        # A squared wave field
        x_list = np.linspace(0, self.period, num_points, endpoint=False)
        y_list = np.copy(x_list)
        x_mesh, y_mesh = np.meshgrid(x_list, y_list)

        kx_list = np.arange(1, self.num_mode + 1) * self.base
        ky_list = np.arange(-self.num_mode, self.num_mode + 1) * self.base
        
        def wrapper(i_sub, t):
            snapshot = np.zeros_like(y_mesh)
            for i in i_sub:
                for j, ky in enumerate(ky_list):
                    w = np.sqrt(g * np.sqrt(kx_list[i]**2 + ky**2))
                    snapshot += (self.Amplitude[i,j] * 
                                np.cos(kx_list[i] * x_mesh
                                     + ky * y_mesh - w * t * self.Tp
                                     + self.random_phase[i,j]))
            return snapshot
    
        i_sub_all = np.array_split(np.arange(self.num_mode), num_p)
        snapshot_series = []
        for t in t_series:
            snapshot = np.sum(Parallel(n_jobs = num_p) 
                            (delayed(wrapper)(i_sub, t) for i_sub in i_sub_all),
                            axis=0)
            snapshot_series.append(snapshot)
        
        return (x_mesh, y_mesh), snapshot_series
    
    
    def generate_envelope(self, num_points=32*8, num_p=6, 
                          cut=32*2, t_series=[0]):
                              
        x_list = np.linspace(0, self.period, num_points, endpoint=False)
        y_list = np.copy(x_list)
        x_mesh, y_mesh = np.meshgrid(x_list, y_list)
        if cut==None:
            cut = self.num_mode
        kx_list = np.arange(1, self.num_mode + 1) * self.base
        ky_list = np.arange(-self.num_mode, self.num_mode + 1) * self.base

        def wrapper(i_sub, t):
            snapshot = np.zeros_like(y_mesh, dtype=complex)
            for i in i_sub:
                for j in range(self.num_mode - cut,
                               self.num_mode + cut + 1):
                    w = np.sqrt(g * np.sqrt((kx_list[i])**2 + 
                                ky_list[j]**2))
                    snapshot += (self.Amplitude[i,j] * 
                                np.exp(1j *((kx_list[i] - self.kp) * x_mesh 
                                     + ky_list[j] * y_mesh 
                                     - w * t * self.Tp
                                     + self.random_phase[i,j])))
            return snapshot
        
        i_sub_all = np.array_split(np.arange(cut), num_p)
        snapshot_series = []
        for t in t_series:
            snapshot = np.sum(Parallel(n_jobs = num_p) 
                            (delayed(wrapper)(i_sub, t) for i_sub in i_sub_all),
                            axis=0)
            snapshot_series.append(snapshot)

        return (x_mesh, y_mesh), np.abs(snapshot_series)

    def _spreading(self, angle_list, theta): 
        theta_list = np.ones_like(angle_list) * theta
        value_list = np.where(abs(angle_list) < theta_list / 2, 
                            2 / theta * np.cos(pi * angle_list / theta)**2,
                            0)
        return value_list