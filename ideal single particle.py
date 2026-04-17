import numpy as np
import matplotlib.pyplot as plt

# Step 1

# Time axis
f_rev = 1e6                # sampling freq: 1 MHz
T_rev = 1.0 / f_rev        # time of complete round : 1 micro s
num_turns = 5              # number of turns in the loop


pass_times = np.arange(1, num_turns + 1) * T_rev

#when we know the features of the single beam, we can replace line 15
#with the exact data. for now it only shows 1 for all.

signal_amplitude = np.ones(num_turns)

#---------------------------------------------

plt.figure(figsize=(7, 5))

plt.stem(pass_times * 1e6, signal_amplitude, linefmt='magenta', markerfmt='ko', basefmt="k-")
plt.title("Single Ideal Particle Passing Pick-up")
plt.xlabel("Time [micro s]")
plt.ylabel("Pick-up Signal")

plt.xlim(0, (num_turns + 1) * T_rev * 1e6)
plt.ylim(0, 2.0)

plt.xticks(pass_times * 1e6)
plt.grid(True, linestyle='--', alpha=0.8)
#

plt.show()