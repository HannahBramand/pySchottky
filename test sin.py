import numpy as np
import scipy
import matplotlib.pyplot as plt

time = np.linspace(0, 10, 1000)
amplitude = 1.0
frequency = 1.0
    
    
signal = amplitude * np.sin(2* np.pi * frequency * time)

plt.plot(time, signal)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")

plt.grid(True)

plt.show()