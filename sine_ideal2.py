import numpy as np
import matplotlib.pyplot as plt

total_time=1e-3
f_rev = 1e6     
#f_s=10e6           
T_rev = 1.0 / f_rev        
#num_turns =  100           
#dt = 1.0 / f_s

W_A = 1.667
A0=0.0
dA_A = 1.0

num_turns = int(total_time * f_rev)
print(num_turns)

t_continuous = np.linspace(0, num_turns * T_rev, 5000)
smooth_amplitude = A0 + dA_A * np.sin(W_A * (t_continuous * 1e6))

pass_times = np.arange(1, num_turns + 1) * T_rev
signal_amplitude = A0+ dA_A * np.sin(W_A * (pass_times * 1e6))


plt.figure(figsize=(10, 5))

plt.plot(t_continuous * 1e6, smooth_amplitude, color='magenta', linestyle='-', linewidth=1.5)

plt.plot(pass_times * 1e6, signal_amplitude, linestyle='none', 
         marker='o', markerfacecolor='black', markeredgecolor='black')

plt.title("Sine plot of Single Ideal Particle Passing Pick-up")
plt.xlabel(r"Time [$\mu$s]")
plt.ylabel("Amplitude")

plt.xlim(0, 100)
plt.ylim(-1.5, 1.5)

plt.xticks(np.arange(0, 100, 5))

plt.axhline(0, color='black', linewidth=1)

plt.grid(True, linestyle='--', alpha=0.8)

plt.tight_layout()
plt.show()