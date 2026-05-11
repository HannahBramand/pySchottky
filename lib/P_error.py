import numpy as np

def get_momentum_data(f_rev, num_turns, slip_fac_eta, dp_p):
    T_rev = 1.0 / f_rev
    T_momentum = T_rev * (1 + (slip_fac_eta * dp_p))
    
    pass_times = np.arange(1, num_turns + 1) * T_rev
    pass_times2 = np.arange(1, num_turns + 1) * T_momentum
    
    return pass_times, pass_times2