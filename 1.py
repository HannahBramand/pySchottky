import numpy as np
import matplotlib.pyplot as plt

# Step 1

# Time axis
f_rev = 1e6                # sampling freq: 1 MHz
T_rev = 1.0 / f_rev        # time of complete round : 1 micro s
num_turns =  20             # number of turns in the loop
frac_tune =   0.66         #fractional tune

Q_b = np.arange(1,num_turns + 1) * 2 * np.pi * frac_tune
#print("betatron phase is :" , Q_b)


pass_times = np.arange(1, num_turns + 1) * T_rev

signal_amplitude = np.ones(num_turns)

signal_amplitude2 = np.cos(Q_b)

#---------------------------------------------

#--------------------------------
plt.figure(figsize=(10, 7))

plt.subplot(2,1,1)
plt.stem(pass_times *1e6, signal_amplitude, linefmt='g-')
plt.plot(pass_times *1e6, signal_amplitude ,color='green' , linestyle='--' , alpha=0.5)
plt.title("Single Ideal Particle (No Betatron Oscillation)")
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(pass_times *1e6, signal_amplitude2, linefmt='b-')
plt.plot(pass_times *1e6, signal_amplitude2 ,color='blue' , linestyle='--' , alpha=0.5)
plt.title("Particle with Fractional Tune")
plt.xlabel("time micro s")
plt.ylabel("Amplitude")
plt.grid(True)


plt.tight_layout()
plt.show()