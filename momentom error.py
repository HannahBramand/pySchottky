import numpy as np
import matplotlib.pyplot as plt
from lib.P_error import get_momentum_data

f_rev = 1e6                
num_turns = 20                          
slip_fac_eta = 0.2
dp_p = 0.1


signal_amplitude = np.ones(num_turns)

ideal, shifted = get_momentum_data(f_rev, num_turns, slip_fac_eta, dp_p)

#--------------------------------
plt.stem(pass_times *1e6, signal_amplitude, linefmt='r-', label="ideal particle")
plt.plot(pass_times *1e6, signal_amplitude ,color='red' , linestyle='--' , alpha=0.5)
plt.stem(pass_times2 *1e6, signal_amplitude, linefmt='b-', label=r"off-momentum particle $\delta$="+"%g [x10$^{-3}$]"%(momentom_error*1E3))
plt.plot(pass_times2 *1e6, signal_amplitude ,color='blue' , linestyle='--' , alpha=0.5)

plt.title("Single Particle ")
plt.grid(True)
plt.xlabel(r"time [$\mu$s]")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
