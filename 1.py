import numpy as np
import matplotlib.pyplot as plt

# پارامترهای اصلی
f_rev = 1e6                # فرکانس چرخش: 1 مگاهرتز
T_rev = 1.0 / f_rev        # زمان یک دور کامل: 1 میکروثانیه
num_turns = 10             # تعداد دورها برای نمایش (برای وضوح بهتر)

# تنظیمات نرخ نمونه‌برداری (Sampling Rate)
# برای اینکه صفرها را ببینیم، نرخ نمونه‌برداری باید بسیار بیشتر از f_rev باشد
fs = 100e6                 # نرخ نمونه‌برداری دستگاه: 100 مگاهرتز
dt = 1.0 / fs              # گام زمانی

# ایجاد محور زمانی کل (مثلاً برای 10 میکروثانیه)
total_time = num_turns * T_rev
t = np.arange(0, total_time, dt)

# ایجاد سیگنال خروجی (در ابتدا همه صفر)
sensor_amplitude = np.zeros_like(t)

# تعیین لحظات دقیق عبور (T_rev, 2*T_rev, ...)
# پیدا کردن ایندکس‌هایی که زمان در آن‌ها مضربی از T_rev است
for n in range(1, num_turns + 1):
    arrival_time = n * T_rev
    # پیدا کردن نزدیک‌ترین نقطه در آرایه زمانی به زمان واقعی عبور
    idx = np.argmin(np.abs(t - arrival_time))
    sensor_amplitude[idx] = 1.0

plt.figure(figsize=(12, 5))
plt.plot(t * 1e6, sensor_amplitude, color='blue', linewidth=1)

plt.title("Sensor Signal (Ideal Impulse at each $T_{rev}$)")
plt.xlabel("Time [$\mu s$]")
plt.ylabel("Amplitude")
plt.ylim(-0.2, 1.2)
plt.grid(True, linestyle=':', alpha=0.7)

tick_locs = np.arange(1, num_turns + 1) * T_rev * 1e6
plt.xticks(tick_locs)

plt.tight_layout()
plt.show()