import numpy as np
from wave import LinearRandomWave
from joblib import Parallel, delayed


def detect_batch(num_seeds, num_p, kws_wave_spectrum, 
                      threshold=6, zero_gap=2):
    statistics_list = Parallel(n_jobs=num_p)(delayed(detect_single)
                                 (seed, kws_wave_spectrum, threshold, zero_gap) 
                                 for seed in range(num_seeds))
    return statistics_list


def detect_single(seed, kws_wave_spectrum, threshold=6, zero_gap=2):
    
    # generate wave field
    wave = LinearRandomWave(**kws_wave_spectrum)
    wave.prepare_wave(seed)
    t_eval = np.arange(0, wave.period, 0.1) 
    elevation = wave.generate_wave(t_eval)
    
    # generate single waves
    zero_crossing = np.where(np.diff(np.sign(elevation)))[0] + 1 
    if elevation[-1] * elevation[0] < 0:   # consider the ending point situation
        zero_crossing = np.append(zero_crossing, len(elevation)-1)    

    A = [np.max(abs(elevation[zero_crossing[i]:zero_crossing[i+2]])) 
              for i in range(0, len(zero_crossing)-2, 2)]
    
    T = [t_eval[zero_crossing][i+2] - t_eval[zero_crossing][i] 
                  for i in range(0, len(zero_crossing)-2, 2)]
    
    S = [t_eval[zero_crossing[i]] for i in range(0, len(zero_crossing), 2)]
        
    A.append(max(np.max(abs(elevation[0: zero_crossing[0]])),
                        np.max(abs(elevation[zero_crossing[-2]:]))))
    T.append((t_eval[-1] - t_eval[zero_crossing][-2]
             + t_eval[zero_crossing][0]))
    
    # Generate the wave groups based on those AST results 
    num_zero = 0    # The tolerance of single waves below the threshold 
    zero_flag = False


    for i in range(0, len(A)):
        if A[i] >= threshold:
            num_zero = 0
            if zero_flag == True:
                starting_index = i
                break
        else: # True if find consective num zero_gap zeros
            num_zero += 1
            if num_zero >= zero_gap:
                zero_flag = True

    num_zero = 0
    num_length = 0
    result = []
            
    # Need to find the starting index. 
    for i in range(starting_index, starting_index + len(A)):
        if A[i % len(A)] >= threshold:     
            num_length += 1
            num_zero = 0         
        else:
            num_zero += 1        
            if num_length > 0: 
                num_length += 1
                # where one group ends
                if num_zero == zero_gap:
                    # results: [[starting index, how many single waves]]
                    result.append([(i - num_length + 1) % len(A), 
                                   num_length - zero_gap])
                    num_length = 0
    
    statistics = []
    for group in result:
        starting_point = S[group[0]]
        if (group[0] + group[1]) <= len(A):
            max_amp = max(A[group[0]:group[0] + group[1]])
            total_T = sum(T[group[0]:group[0] + group[1]])
        else:
            max_amp = max(max(A[group[0]:]), 
                          max(A[: (group[0] + group[1]) % len(A)]))
            total_T = sum([sum(T[group[0]:]), 
                          sum(T[: (group[0] + group[1]) % len(T)])])
        statistics.append([seed, starting_point, max_amp, total_T, group[1]])
    return statistics
    
    
if __name__ == "__main__":
    num_p = 10
    num_seeds = 10000
    threshold = 6
    zero_gap = 2
    kws_wave_spectrum = {'gamma':3, 'whether_Gaussian':False}
    save_address = 'broadband'
    
    final = detect_batch(num_seeds, num_p, kws_wave_spectrum, 
                         threshold, zero_gap)
    np.save(save_address, np.array(final, dtype=object))