import numpy as np

def get_momentum_data(f_rev, num_turns, slip_fac_eta, dp_p):
    T_rev = 1.0 / f_rev
    T_momenta = [T_rev * (1 + (slip_fac_eta * myDp_p)) for myDp_p in dp_p]
    pass_times = [ np.arange(1, num_turns + 1) * T_momentum for T_momentum in T_momenta ]
    
    return pass_times
