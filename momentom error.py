import numpy as np
import matplotlib.pyplot as plt

# Step 1

# Time axis
f_rev = 1e6                # sampling freq: 1 MHz
T_rev = 1.0 / f_rev        # time of complete round : 1 micro s
num_turns =  20             # number of turns in the loop
frac_tune =   0.66         #fractional tune

slip_fac_eta = 0.1
momentom_error = 0.1

T_momentom = T_rev * (1+ (slip_fac_eta * momentom_error))

pass_times = np.arange(1, num_turns + 1) * T_rev

pass_times2 = np.arange(1 , num_turns +1) * T_momentom

signal_amplitude = np.ones(num_turns)

#--------------------------------
plt.figure(figsize=(10, 7))

plt.subplot(2,1,1)
plt.stem(pass_times *1e6, signal_amplitude, linefmt='g-')
plt.plot(pass_times *1e6, signal_amplitude ,color='green' , linestyle='--' , alpha=0.5)
plt.title("Single Ideal Particle ")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(pass_times2 *1e6, signal_amplitude, linefmt='b-')
plt.plot(pass_times2 *1e6, signal_amplitude ,color='blue' , linestyle='--' , alpha=0.5)
plt.title("momentum error")
plt.xlabel("time micro s")
plt.ylabel("Amplitude")
plt.grid(True)


plt.tight_layout()
plt.show()