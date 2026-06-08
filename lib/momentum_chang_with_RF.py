import numpy as np

def get_momentum_data(f_rev, num_turns, slip_fac_eta, dp_p_list, V_rf=0.005):

    T_rev = 1.0 / f_rev
    all_pass_times = []
    
    for initial_dp in dp_p_list:
        current_dt = 0.0          
        current_dp = initial_dp   
        
        pass_times = []
        
        for n in range(num_turns):
            phase = 2 * np.pi * f_rev * current_dt
            
            current_dp = current_dp - V_rf * np.sin(phase)
            
            current_dt = current_dt + T_rev * slip_fac_eta * current_dp
            
            absolute_time = (n + 1) * T_rev + current_dt
            pass_times.append(absolute_time)
            
        all_pass_times.append(np.array(pass_times))
        
    return all_pass_times