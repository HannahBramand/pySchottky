import numpy as np
import matplotlib.pyplot as plt

f_rev = 1e6                 
T_rev = 1.0 / f_rev         
num_turns = 124            

slip_fac_eta = 0.8          
dp_p_max = 0.8              

f_w = 50e3                  
w = 2 * np.pi * f_w

signal_amplitude = np.ones(num_turns)
pass_times = []

for n in range(num_turns):
    t_ideal = (n + 1) * T_rev
    
    current_dp_p = dp_p_max * np.sin(w * t_ideal)
    
    dt = T_rev * slip_fac_eta * current_dp_p
    
    t_absolute = t_ideal + dt
    pass_times.append(t_absolute)

pass_times = np.array(pass_times)

fig, ax = plt.subplots(figsize=(12, 6))
ax.stem(pass_times * 1e6, signal_amplitude, linefmt="b-", markerfmt="bo")
ax.plot(pass_times * 1e6, signal_amplitude, color="blue", alpha=0.3)

ax.set_title(r"Single Particle with Analytical Momentum Modulation: $p_0 + \frac{\Delta p}{p_0}\sin(\omega t)$")
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel("Amplitude")

plt.tight_layout()
plt.show()