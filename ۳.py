import numpy as np
import matplotlib.pyplot as plt

# پارامترهای فیزیکی
f_rev = 1e6                
T_rev_ideal = 1.0 / f_rev        
num_turns = 20
slip_fac_eta = 0.2        

# پارامترهای نوسان (Synchrotron Oscillation)
A = 0.1                    # دامنه نوسان دلتا پی به پی
nu_s = 0.05                # فرکانس نوسان (چند نوسان در کل دوره؟)

# محاسبه زمان‌های عبور برای یک ذره نوسان‌گر
pass_times = []
current_time = 0

for n in range(1, num_turns + 1):
    # مقدار دلتا پی در این دور خاص
    current_dp_p = A * np.cos(2 * np.pi * nu_s * n)
    
    # زمان چرخش در این دور (وابسته به مقدار لحظه‌ای dp/p)
    T_rev_actual = T_rev_ideal * (1 + (slip_fac_eta * current_dp_p))
    
    current_time += T_rev_actual
    pass_times.append(current_time)

pass_times = np.array(pass_times)
signal_amplitude = np.ones(num_turns)

# رسم پلات
plt.figure(figsize=(12, 6))

# رسم پالس‌های ذره نوسان‌گر (با رنگ سبز طبق سلیقه قبلی)
plt.stem(pass_times * 1e6, signal_amplitude, linefmt='g-', markerfmt='go', label='Oscillating Particle')

# رسم خطوط راهنما برای ذره ایده آل (بدون نوسان) جهت مقایسه
ideal_times = np.arange(1, num_turns + 1) * T_rev_ideal
plt.vlines(ideal_times * 1e6, 0, 0.5, colors='r', linestyles='--', alpha=0.3, label='Ideal Reference')

plt.title(r"Single Particle with Momentum Oscillation ($\Delta p/p$ swinging between -0.1 and 0.1)")
plt.xlabel(r"time [$\mu$s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()