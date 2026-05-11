import numpy as np
import matplotlib.pyplot as plt

# import our lovely libraries
import sys
sys.path.append("./lib")
from lib import Pulses 




f_s = 1e9                  # Sampling frequency: 1 GHz
T_s = 1.0 / f_s            # Sampling time step: 1 ns
t_end = 3e-6               # End time: 3 microseconds (to see a few turns)
time = np.arange(0, t_end, T_s) # Time axis line


f_rev = 1e6               # Revolution frequency: 1 MHz
T_rev = 1.0 / f_rev        # Turn period: 1 microsecond
sigma = 60e-9       # Bunch length (very short pulse)
center = T_rev / 2.0       # Center of the first pulse


signal_ideal = Pulses.GaussianTrain( time, sigma, f_rev, T0=center )

# -------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(time , signal_ideal, color='magenta')

plt.title(f"Ideal Bunched Beam Current (Test 1) \n(Turn Period = {T_rev*1e6} us, Pulse Width sigma = {sigma*1e9:.1f} ns)")
plt.xlabel("Time (s) ")
plt.ylabel("Beam Current")
plt.grid(True, linestyle='--', alpha=1.0)

plt.tight_layout()
plt.show()
