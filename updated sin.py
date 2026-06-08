import os
import numpy as np
import matplotlib.pyplot as plt

total_time = 10e-3      
shape = 'sine'               
f_s = 200e6        
f_rev = 1.2e6
T_rev = 1.0 / f_rev
dt = 1.0 / f_s

f_w = 10e3                   

num_turns = int(total_time * f_rev)
print(num_turns)

slip_fac_eta = 0.3          
dp_p = 0.1                  

pass_times = []
T_new = 0.0

for n in range(num_turns):
    t_ideal = (n + 1) * T_rev  
    
    if shape == 'single point':
        current_dp = dp_p    
    elif shape == 'sine':
        current_dp = dp_p * np.sin(2 * np.pi * f_w * t_ideal)
        
    T_new = T_new + T_rev * slip_fac_eta * current_dp
    
    absolute_time = t_ideal + T_new
    pass_times.append(absolute_time)

pass_times = np.array(pass_times)
print(f"Final T_new (Time slip): {T_new}")

t_digital = np.arange(0, total_time, dt)
sensor_signal = np.zeros_like(t_digital)

indices = np.round(pass_times / dt).astype(int)
indices = indices[indices < len(t_digital)]
sensor_signal[indices] = 1.0

n_fft = len(t_digital)
freqs = np.fft.fftfreq(n_fft, d=dt)
fft_values = np.fft.fft(sensor_signal)
amplitude_spectrum = (np.abs(fft_values) / n_fft) * 2

mask = (freqs >= 0) & (freqs <= 5e6) 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

t_continuous = np.linspace(0, 100e-6, 1000)
p0 = 1.0 
if shape == 'sine':
    momentum_plot = p0 + dp_p * np.sin(2 * np.pi * f_w * t_continuous)
else:
    momentum_plot = p0 + dp_p * np.ones_like(t_continuous)

ax1.plot(t_continuous * 1e6, momentum_plot, color='green', linewidth=2)
ax1.set_title(r"Analytical Momentum: $p(t) = p_0 + \frac{\Delta p}{p_0} \sin(wt)$")
ax1.set_xlabel(r"Time [$\mu$s]")
ax1.set_ylabel("Normalized Momentum")
ax1.grid(True, linestyle=':', alpha=0.7)

time_mask = t_digital   <= 100e-6
ax2.plot(t_digital[time_mask] * 1e6, sensor_signal[time_mask], color='blue')
ax2.set_title(f"Sensor Signal (First 100 $\mu$s) - {shape.upper()}")
ax2.set_xlabel(r"Time [$\mu$s]")
ax2.set_ylabel("Amplitude")
ax2.grid(True, linestyle=':', alpha=0.7)

ax3.plot(freqs[mask] / 1e6, amplitude_spectrum[mask], color='red')
ax3.axvline(f_rev / 1e6 , color='black', linestyle='--', alpha=0.5, label='Ideal $f_{rev}$')
ax3.set_title(f"Frequency Domain (FFT) - Sampling: {f_s/1e6} MHz")
ax3.set_xlabel("Frequency [MHz]")
ax3.set_ylabel("Amplitude")
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.legend()

saved_freqs = freqs[mask]
saved_amps = amplitude_spectrum[mask]
dataset_matrix = np.column_stack((saved_freqs, saved_amps))

safe_shape_name = shape.replace(' ', '_')
filename = f"dataset_FFT_{safe_shape_name}_{int(f_s/1e6)}MHz.csv"

current_directory = os.path.dirname(os.path.abspath(__file__))
full_path = os.path.join(current_directory, filename)

np.savetxt(full_path, dataset_matrix, delimiter=',', header='Frequency_Hz,Amplitude', comments='')
print(f"✅ Dataset successfully saved EXACTLY at: {full_path}")

#-------------------
dataset_time_matrix = np.column_stack((t_digital, sensor_signal))

filename_time = f"dataset_Time_{safe_shape_name}_{int(f_s/1e6)}MHz.csv"
full_path_time = os.path.join(current_directory, filename_time)

np.savetxt(full_path_time, dataset_time_matrix, delimiter=',', header='Time_s,Amplitude', comments='')
print(f"✅ Time Dataset successfully saved EXACTLY at: {full_path_time}")

plt.tight_layout(h_pad=4.0)
plt.show()