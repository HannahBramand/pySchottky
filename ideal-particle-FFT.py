import numpy as np
import matplotlib.pyplot as plt


f_rev = 1.21e6                
T_rev = 1.0 / f_rev        
num_turns = 100         
fs = 100e6                
dt = 1.0 / fs              

t = np.arange(0, num_turns * T_rev, dt)
sensor_amplitude = np.zeros_like(t)

for n in range(1, num_turns):
    arrival_time = n * T_rev
    idx = np.argmin(np.abs(t - arrival_time))
    sensor_amplitude[idx] = 1.0


n_fft = len(t)
freqs = np.fft.fftfreq(n_fft, d=dt)
fft_values = np.fft.fft(sensor_amplitude)

amplitude_spectrum = (np.abs(fft_values) / n_fft) * 2

positive_freq_mask = (freqs >= 0) & (freqs <= 10e6)
freqs_plot = freqs[positive_freq_mask]
fft_plot = amplitude_spectrum[positive_freq_mask]

# --- رسم نمودارها ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ۱. نمودار حوزه زمان (نقاط سبز)
ax1.plot(t * 1e6, sensor_amplitude, 'go', markersize=3, label="Sampled Points")
ax1.set_title("Time Domain Signal (Green Points)")
ax1.set_xlabel("Time [$\mu s$]")
ax1.set_ylabel("Amplitude")
ax1.set_ylim(-0.2, 1.2)
ax1.set_xlim(0, 10) # نمایش ۱۰ میکروثانیه اول برای وضوح
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# ۲. نمودار حوزه فرکانس (Amplitude Spectrum)
ax2.plot(freqs_plot / 1e6, fft_plot, color='blue', linewidth=1.5)
ax2.set_title("Frequency Domain (Amplitude Spectrum)")
ax2.set_xlabel("Frequency [MHz]")
ax2.set_ylabel("Magnitude") # واحد اکنون بر اساس دامنه است
ax2.grid(True, linestyle=':', alpha=0.8)

# خطوط راهنما برای هارمونیک‌ها
for i in range(1, 11):
    ax2.axvline(i * f_rev / 1e6, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()