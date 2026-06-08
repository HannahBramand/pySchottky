import numpy as np
import matplotlib.pyplot as plt
from lib.momentum_chang_with_RF import get_momentum_data

f_rev = 1e6                
num_turns = 124
slip_fac_eta = 0.2        
dp_p = [0.0, 0.0, 0.0]
myColors = ["r", "b", "g", "c", "m", "k"]

signal_amplitude = np.ones(num_turns)      

# فراخوانی تابع که حالا RF در آن روشن است
pass_times = get_momentum_data(f_rev, num_turns, slip_fac_eta, dp_p)

#--------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

for ii in range(len(pass_times)):
    label_str = f"$\Delta p/p_0$ = {dp_p[ii]}"
    
    # استفاده از label در stem برای نمایش در راهنما
    ax.stem(pass_times[ii] * 1e6, signal_amplitude, linefmt="%s-" % (myColors[ii]), label=label_str)
    ax.plot(pass_times[ii] * 1e6, signal_amplitude, color=myColors[ii], alpha=0.3)

ax.set_title("Time Domain Signal with RF ON (Synchrotron Oscillations)")
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel(r"Time [$\mu s$]")
ax.set_ylabel("Amplitude")
#ax.legend(loc='upper right')

plt.tight_layout()
plt.show()