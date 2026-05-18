import numpy as np
import matplotlib.pyplot as plt
from lib.P_error import get_momentum_data

f_rev = 1e6                
num_turns = 124
slip_fac_eta = 0.2        
dp_p = [0.0, 0.1,-0.1]
myColors=["r","b","g","c","m","k"]


signal_amplitude = np.ones(num_turns)      

pass_times = get_momentum_data(f_rev, num_turns, slip_fac_eta, dp_p)

#--------------------------------
for ii in range(len(pass_times)):
    plt.stem(pass_times[ii]*1e6, signal_amplitude, linefmt="%s-"%(myColors[ii]))
    plt.plot(pass_times[ii]*1e6, signal_amplitude, color=myColors[ii])
# plt.stem(ideal *1e6, signal_amplitude, linefmt='r-', label="ideal particle")
# plt.plot(ideal *1e6, signal_amplitude ,color='red' , linestyle='--' , alpha=0.5)
# plt.stem(shifted *1e6, signal_amplitude, linefmt='b-', label=r"off-momentum particle $\delta$="+"%g [x10$^{-3}$]"%(dp_p*1E3))
# plt.plot(shifted *1e6, signal_amplitude ,color='blue' , linestyle='--' , alpha=0.5)

plt.title("Three individual particles (No RF)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel(r"time [$\mu$s]")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
