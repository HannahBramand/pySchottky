import numpy as np
import matplotlib.pyplot as plt

# پارامترهای فیزیکی
f_rev = 1e6                
T_rev_ideal = 1.0 / f_rev        
num_turns = 20             # تعداد دورها برای وضوح نمایش
slip_fac_eta = 0.2        
A = 0.1                    # دامنه نوسان دلتا پی به پی
nu_s = 0.1                 # فرکانس نوسان

# پارامترهای شکل گاوسی پالس
sigma_t = 0.05e-6          # پهنای زمانی پالس گاوسی (مثلاً 50 نانو ثانیه)
time_resolution = 1000     # تعداد نقاط در هر دور برای رسم تمیز

# ایجاد محور زمانی بسیار دقیق
t_total = np.linspace(0, num_turns * T_rev_ideal * 1.1, num_turns * time_resolution)
amplitude = np.zeros_like(t_total)

# تابع گاوسی
def gaussian(t, center, sigma):
    return np.exp(-((t - center)**2) / (2 * sigma**2))

# محاسبه زمان‌های عبور با نوسان و اضافه کردن شکل گاوسی
current_arrival_time = 0
arrival_times = []

for n in range(1, num_turns + 1):
    dp_p = A * np.cos(2 * np.pi * nu_s * n)
    T_rev_actual = T_rev_ideal * (1 + (slip_fac_eta * dp_p))
    current_arrival_time += T_rev_actual
    arrival_times.append(current_arrival_time)
    
    # اضافه کردن شکل گاوسی به سیگنال کلی در این لحظه
    amplitude += gaussian(t_total, current_arrival_time, sigma_t)

# رسم نمودار
plt.figure(figsize=(12, 5))
plt.plot(t_total * 1e6, amplitude, color='green', linewidth=2, label='Gaussian Pulse Shape')

# مشخص کردن مرکز پالس‌ها با نقطه برای مقایسه
plt.scatter(np.array(arrival_times) * 1e6, np.ones(num_turns), color='red', s=20, label='Pulse Center')

plt.title("Sensor Signal with Gaussian Pulse Shapes & Momentum Oscillation")
plt.xlabel("Time [$\mu s$]")
plt.ylabel("Amplitude (A.U.)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()