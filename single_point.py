import numpy as np
import matplotlib.pyplot as plt

total_time= 1e-3         
shape = 'single point'      
f_s= 200e6               
f_rev = 1.2e6 
T_rev = 1.0 / f_rev
dt = 1.0 / f_s

#num_turns=
num_turns = int(total_time * f_rev)

slip_fac_eta = 0.3          
dp_p = 0.1                  

pass_times = []
T_new = 0.0

for n in range(num_turns):
    if shape == 'single point':
        current_dp = dp_p    
        T_new = T_new + T_rev * slip_fac_eta * current_dp
    
    absolute_time = (n + 1) * T_rev + T_new
    pass_times.append(absolute_time)

pass_times = np.array(pass_times)

print(T_new)

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

time_mask = t_digital <= 50e-6

ax1.set_title(f"Time Domain (First 50 $\mu$s) - {shape.upper()}")
ax1.plot(t_digital[time_mask] * 1e6, sensor_signal[time_mask], color='blue')
#ax1.plot(t_digital * 1e6, sensor_signal, color='blue', marker='.', linestyle='none')
ax1.set_xlabel("Time [$\mu$s]")
ax1.set_ylabel("Amplitude")
ax1.grid(True, linestyle=':', alpha=0.7)

#ax1.plot(t_digital * 1e6, sensor_signal, color='blue')
#ax1.set_title(f"Time Domain (Full {total_time * 1000} ms) - {shape.upper()}")
#ax1.set_xlabel(r"Time [$\mu$s]")
#ax1.set_ylabel("Amplitude")
#ax1.grid(True, linestyle=':', alpha=0.7)

ax2.plot(freqs[mask] / 1e6, amplitude_spectrum[mask], color='red')
ax2.axvline(f_rev / 1e6 , color='black', linestyle='--', alpha=0.5, label='Ideal $f_{rev}$')
ax2.set_title(f"Frequency Domain (FFT) - Sampling: {f_s} MHz")
ax2.set_xlabel("Frequency [MHz]")
ax2.set_ylabel("Amplitude")
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

# ==========================================
saved_freqs = freqs[mask]
saved_amps = amplitude_spectrum[mask]

dataset_matrix = np.column_stack((saved_freqs, saved_amps))

safe_shape_name = shape.replace(' ', '_')
filename = f"dataset_FFT_{safe_shape_name}_{int(f_s/1e6)}MHz.csv"

np.savetxt(filename, dataset_matrix, delimiter=',', header='Frequency_Hz,Amplitude', comments='')

print(f"✅ Dataset successfully saved as: {filename}")


plt.tight_layout()
plt.show()

