import os
import numpy as np
import matplotlib.pyplot as plt

total_time = 1e-3
f_rev = 1.2e6               
T_rev = 1.0 / f_rev
f_s = 10e6
dt = 1.0 / f_s

#num_turns = 100         
f_A = 5.0/30
A0 = 0.0
dA_A = 1.0

num_turns = int(total_time * f_rev)

pass_times = np.arange(0, num_turns + 1) * T_rev
signal_amplitude = A0 + dA_A * np.sin(2 * np.pi * f_A * f_rev * pass_times)

print(num_turns)
#----
n_fft = len(signal_amplitude)
freqs = np.fft.fftfreq(n_fft, d=dt)
fft_values = np.fft.fft(signal_amplitude)

amplitude_spectrum = (np.abs(fft_values) / n_fft) * 2

freqs_shifted = np.fft.fftshift(freqs)
#amps_shifted = np.fft.fftshift(amplitude_spectrum)

#---------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(pass_times * 1e6, signal_amplitude, color='magenta', linestyle='-', linewidth=1.5)
ax1.plot(pass_times * 1e6, signal_amplitude, linestyle='none', 
         marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4)

ax1.set_title(r"Time Domain: Transverse Oscillation ($\mu$s)")
ax1.set_xlabel(r"Time [$\mu$s]")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0, 500)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(freqs/ 1e6, amplitude_spectrum, '.--', color='red', linewidth=1.5)

ax2.axvline(0, color='black', linewidth=1, linestyle='-')

ax2.set_title("Frequency Domain (FFT) - Full Spectrum (Positive & Negative)")
ax2.set_xlabel("Frequency [MHz]")
ax2.set_ylabel("Amplitude")
ax2.grid(True, linestyle='--', alpha=0.7)


export_data = np.column_stack((freqs / 1e6, signal_amplitude))
current_folder = os.path.dirname(os.path.abspath(__file__))
full_save_path = os.path.join(current_folder, 'A_FFT_data.csv')
np.savetxt(full_save_path, export_data, delimiter=',', header='Frequency (MHz),Amplitude', comments='')
print(f"file has been saved. \n{full_save_path}")



plt.tight_layout(h_pad=3.0)
plt.show()