import numpy as np
import matplotlib.pyplot as plt
from lib.P_error import get_momentum_data

f_rev = 1e6               
num_turns = 124          
slip_fac_eta = 0.2         
dp_p = [0.0, 0.1, -0.1]    
myColors = ["r", "b", "g", "c", "m", "k"]


signal_amplitude = np.ones(num_turns)

pass_times = get_momentum_data(f_rev, num_turns, slip_fac_eta, dp_p)

t0 = pass_times[0]

# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))
# ------------------------------------------------------------------

for ii in range(len(pass_times)):
    
    t = pass_times[ii]
    ax1.stem(t * 1e6, signal_amplitude, linefmt="%s-"%(myColors[ii]))
    #ax1.plot(t * 1e6, signal_amplitude, color=myColors[ii])
       
    dt = t - t0
    
    dp_p0 = np.full(num_turns, dp_p[ii])
    ax2.scatter(dt * 1e6, dp_p0, color=myColors[ii])
    #ax2.plot(dt * 1e6, dp_p0, color=myColors[ii], alpha=0.3, linestyle='--')


ax1.set_title("Time Domain Signal for Three individual particles (No RF)")
ax1.set_ylabel("Amplitude")

ax1.set_xlim(0, (num_turns + 5) * (1.0 / f_rev) * (1 + slip_fac_eta * 0.1) * 1e6)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_title("Longitudinal Phase Space: Δp/p₀ vs Δt")
ax2.set_xlabel(r"Time Deviation ($\Delta t$) [$\mu s$]")
ax2.set_ylabel(r"Momentum Deviation ($\Delta p / p_0$)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()


plt.tight_layout()
plt.show()