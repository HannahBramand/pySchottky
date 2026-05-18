import numpy as np
import matplotlib.pyplot as plt

# پارامترها
f_rev = 1e6                
T_rev = 1.0 / f_rev        
num_turns =  20          
fs =  50e6               # نرخ نمونه‌برداری بالا برای مشاهده فواصل صفر
dt = 1.0 / fs              

t = np.arange(0, num_turns * T_rev + dt, dt)
sensor_amplitude = np.zeros_like(t)

for n in range(1, num_turns + 1):
    arrival_time = n * T_rev
    idx = np.argmin(np.abs(t - arrival_time))
    sensor_amplitude[idx] = 1.0

plt.figure(figsize=(12, 5))

# تغییر از خط به نقطه
# 'ko' یعنی رنگ مشکی (k) و شکل دایره (o)
plt.plot(t * 1e6, sensor_amplitude, 'go', markersize=4, label="Sampled Points")

plt.title("Sensor Signal - Discrete Sampling Points")
plt.xlabel("Time [$\mu s$]")
plt.ylabel("Amplitude")
plt.ylim(-0.2, 1.2)
plt.grid(True, linestyle=':', alpha=0.7)

tick_locs = np.arange(1, num_turns + 1) * T_rev * 1e6
plt.xticks(tick_locs)

plt.legend()
plt.tight_layout()
plt.show()