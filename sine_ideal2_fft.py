import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

total_time=1e-3
f_rev = 1.1e6 #  [Hz]              
T_rev = 1.0 / f_rev # [s] 
f_s = 10e6
dt = 1.0 / f_s      
#num_turns =10000

num_turns = int(total_time * f_rev)
print(num_turns)

f_A = 5.0/30 # []
A0=0.0
dA_A = 1.0

t_continuous = np.linspace(0, num_turns, (num_turns*10+1))*T_rev # max included [s]
smooth_amplitude = A0+ dA_A * np.sin(2 * np.pi * f_A * f_rev * t_continuous  )

pass_times = np.arange(0, num_turns + 1) * T_rev # max excluded [s]
signal_amplitude = A0+ dA_A * np.sin(2 * np.pi * f_A  * f_rev * pass_times ) # [s]

n_fft = len(signal_amplitude)
freqs = fft.rfftfreq(n_fft , 1/f_s)
# print(freqs)
fft_values = fft.rfft(signal_amplitude)
 

amplitude_spectrum = (abs(fft_values) / n_fft) * 2

#freqs_shifted = np.fft.fftshift(freqs)
# amps_shifted = np.fft.fftshift(amplitude_spectrum)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

#ax1.vlines(x=pass_times * 1e6, ymin=0, ymax=signal_amplitude, 
 #          color='blue', linestyle='-', alpha=0.5, linewidth=1.5)

ax1.plot(t_continuous * 1e6, smooth_amplitude, '.-', color='magenta', linewidth=1.5)
ax1.plot(pass_times * 1e6, signal_amplitude, linestyle='none', 
         marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4)

ax1.set_title(r"Time Domain: Transverse Oscillation (Zoomed on first 500 $\mu$s)")
ax1.set_xlabel(r"Time [$\mu$s]")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0, 500)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(freqs * 1e-6, amplitude_spectrum, '.-', color='red', linewidth=1.5)

ax2.axvline(0, color='black', linewidth=1, linestyle='-')

ax2.set_title("Frequency Domain (FFT) - Full Spectrum (Positive & Negative)")
ax2.set_xlabel("Frequency [MHz]")
ax2.set_ylabel("Amplitude")
ax2.grid(True, linestyle='--', alpha=0.7)

export_data = np.column_stack((pass_times * 1e6, signal_amplitude))
current_folder = os.path.dirname(os.path.abspath(__file__))
full_save_path = os.path.join(current_folder, 'A_data.csv')
np.savetxt(full_save_path, export_data, delimiter=',', header='Time (s),Amplitude', comments='')
print(f"file has been saved. \n{full_save_path}")


plt.tight_layout(h_pad=4.0)
plt.show()