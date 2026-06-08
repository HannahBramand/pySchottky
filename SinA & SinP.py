import os
import numpy as np
import matplotlib.pyplot as plt

total_time = 1e-3
f_rev = 1e6                
T_rev = 1.0 / f_rev        
f_s = 10e6
dt = 1.0 / f_s

#num_turns = 124      
num_turns = int(total_time * f_rev)
print("number of turns: " , num_turns)

f_A = 5.0/30
A0 = 0.0
dA_A = 1.0

f_p = 1e-3
dp_p = 0.01
slip_fac_eta = 0.5


pass_times = []
T_new = 0.0

for n in range(num_turns):
    t_ideal = (n + 1) * T_rev
    
    current_dp_p = dp_p * np.sin(2 * np.pi * f_p  * f_rev* t_ideal )
    T_new = T_new + T_rev * slip_fac_eta * current_dp_p
    
    t_absolute = t_ideal + T_new
    pass_times.append(t_absolute)

pass_times = np.array(pass_times)

signal_amplitude = A0 + dA_A * np.sin(2 * np.pi * f_A  * f_rev* pass_times)

#---------------
n_fft = len(signal_amplitude)
freqs = np.fft.fftfreq(n_fft, d=dt)
fft_values = np.fft.fft(signal_amplitude)

amplitude_spectrum = (np.abs(fft_values) / n_fft) * 2

freqs_shifted = np.fft.fftshift(freqs)
#amps_shifted = np.fft.fftshift(amplitude_spectrum)
#-------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.vlines(x=pass_times * 1e6, ymin=0, ymax=signal_amplitude, 
           color='blue', linestyle='-', alpha=0.5, linewidth=1.5)

ax1.plot(pass_times * 1e6, signal_amplitude, color='magenta', 
         linestyle='-', linewidth=1.5, alpha=0.5, label="Sampled Path")

ax1.plot(pass_times * 1e6, signal_amplitude, linestyle='none', 
         marker='o', markerfacecolor='black', markeredgecolor='black', 
         zorder=5, markersize=4, label="Sampled by Pick-up")

ax1.set_title(r"Combined Dynamics: Transverse Oscillation & Longitudinal Time Slip")
ax1.set_xlabel(r"Time [$\mu$s]")
ax1.set_ylabel("Amplitude")

ax1.set_xlim(0, 500 )
ax1.set_ylim(-1.5, 1.5) 
ax1.set_xticks(np.arange(0, 500 , 10))
ax1.axhline(0, color='black', linewidth=1)

ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

#------------

ax2.plot(freqs / 1e6, amplitude_spectrum , '.-', color='red', linewidth=1.5)

ax2.axvline(0, color='black', linewidth=1, linestyle='-')

ax2.set_title("Frequency Domain (FFT) - Full Spectrum (Positive & Negative)")
ax2.set_xlabel("Frequency [MHz]")
ax2.set_ylabel("Amplitude")
ax2.grid(True, linestyle='--', alpha=0.7)

#--------------
export_data = np.column_stack((freqs / 1e6, signal_amplitude))
current_folder = os.path.dirname(os.path.abspath(__file__))
full_save_path = os.path.join(current_folder, 'SinA_&_SinP_FFT_data.csv')
np.savetxt(full_save_path, export_data, delimiter=',', header='Frequency (MHz),Amplitude', comments='')
print(f"file has been saved. \n{full_save_path}")

plt.tight_layout(h_pad=4.0)
plt.show()