import os
import numpy as np
import matplotlib.pyplot as plt

total_time = 1e-3
f_rev = 1.1e6 #  [Hz]              
T_rev = 1.0 / f_rev # [s] 
f_s = 10e6
dt = 1.0 / f_s      

num_turns = int(total_time * f_rev)
print(f"Number of turns: {num_turns}")

f_A = 5./30 # []
A0 = 0.0
dA_A = 1.0

t_continuous = np.linspace(0, num_turns, (num_turns*10+1))*T_rev 
smooth_amplitude = A0+ dA_A * np.sin(2 * np.pi * f_A  * (t_continuous * 1e6) )

pass_times = np.arange(0, num_turns + 1) * T_rev 
signal_amplitude = A0+ dA_A * np.sin(2 * np.pi * f_A  * (pass_times * 1e6) ) 

n_fft = len(signal_amplitude)
freqs = np.fft.fftfreq(n_fft, d=dt)
fft_values = np.fft.fft(signal_amplitude)
amplitude_spectrum = (np.abs(fft_values) / n_fft) * 2

# ==========================================
# تغییرات اعمال شده: جداسازی فرکانس‌های مثبت
# ==========================================
pos_mask = freqs >= 0  # پیدا کردن ایندکس فرکانس‌های مثبت
freqs_pos = freqs[pos_mask]
amplitude_pos = amplitude_spectrum[pos_mask]
# ==========================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# نمودار اول (حوزه زمان)
ax1.plot(t_continuous * 1e6, smooth_amplitude, '.-', color='magenta', linewidth=1.5)
ax1.plot(pass_times * 1e6, signal_amplitude, linestyle='none', 
         marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4)

ax1.set_title(r"Time Domain: Transverse Oscillation (Zoomed on first 100 $\mu$s)")
ax1.set_xlabel(r"Time [$\mu$s]")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0, 500)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, linestyle='--', alpha=0.7)

# نمودار دوم (حوزه فرکانس - فقط مثبت)
ax2.plot(freqs_pos / 1e6, amplitude_pos, '.-', color='red', linewidth=1.5) # استفاده از داده‌های مثبت

ax2.set_xlim(left=0) # اطمینان از اینکه محور X از صفر شروع شود
ax2.set_title("Frequency Domain (FFT) - Positive Frequencies Only")
ax2.set_xlabel("Frequency [MHz]")
ax2.set_ylabel("Amplitude")
ax2.grid(True, linestyle='--', alpha=0.7)

# اصلاح بخش ذخیره داده‌ها: ذخیره دامنه فرکانس به جای دامنه سیگنال زمان
export_data = np.column_stack((freqs_pos / 1e6, amplitude_pos)) 
current_folder = os.path.dirname(os.path.abspath(__file__))
full_save_path = os.path.join(current_folder, 'SinA_FFT_data.csv')
np.savetxt(full_save_path, export_data, delimiter=',', header='Frequency (MHz),Amplitude', comments='')
print(f"File has been saved.\n{full_save_path}")

plt.tight_layout(h_pad=4.0)
plt.show()