import numpy as np

def GaussianTrain(time,sigma,freq,T0=0.0,AA=None,signal=None):
    if (signal is None): signal=np.zeros_like(time)
    if (AA is None): AA=sigma*np.sqrt(2.0*np.pi)
    #
    TT = 1.0 / freq 
    num_pulses = int((np.max(time)-np.min(time)) / TT) + 1
    for i in range(num_pulses):
        pulse_center = T0 + i * TT
        signal += AA / (sigma*np.sqrt(2.0*np.pi)) * np.exp(-0.5 * ((time - pulse_center) / sigma)**2)
    return signal

    
    
