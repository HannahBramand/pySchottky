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

plt.figure(figsize=(10, 7))

#plot ideal
plt.subplot(2,1,1)
plt.stem(ideal * 1e6, signal_amplitude, linefmt='g-', label='Ideal')
plt.plot(ideal * 1e6, signal_amplitude, color='green', linestyle='--', alpha=0.5)
plt.title("Single Ideal Particle")
plt.ylabel("Amplitude")
plt.grid(True)

#plot dp_p error
plt.subplot(2,1,2)
plt.stem(shifted * 1e6, signal_amplitude, linefmt='b-', label='With Error')
plt.plot(shifted * 1e6, signal_amplitude, color='blue', linestyle='--', alpha=0.5)
plt.title("Momentum Error Effect")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()