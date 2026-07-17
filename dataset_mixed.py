import numpy as np
import re
import os
import gc

# ******************************************************
#                   1. PARAMETERS
# ******************************************************

GAMMA_T = 1.971            # Transition Gamma for CNAO
F_A = 5.0 / 3.0            # Fractional Betatron Tune (Proton)
DP_P_MAX = 1e-3            # Momentum spread
DA_A_MAX = 0.1             # Transverse modulation depth
F_W = 10e3                 # Longitudinal modulation frequency
SIGMA_T = 70e-9            # Pulse width

TOTAL_TIME = 1e-3          # 1 ms simulation
F_S = 200e6                # Sampling frequency (200 MHz is safe for RAM and high resolution)
DT = 1.0 / F_S
OUTPUT_DIR = "ML_Datasets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ******************************************************
#                  2. Extracting Data
# ******************************************************

def parse_proton_file(filepath):
    print(f"Reading physics data from {filepath}...")
    with open(filepath, 'r') as f:
        text = f.read()

    gammas = re.findall(r'gamma\s*=\s*([0-9.]+)', text)
    energies = re.findall(r'energy\[MeV/u\]\s*=\s*([0-9.]+)', text)
    frequencies = re.findall(r'frequency\[Hz\]\s*=\s*([0-9.e+]+)', text)

    dataset_params = []
    for g, e, f_rev in zip(gammas, energies, frequencies):
        dataset_params.append({
            'energy': float(e),
            'gamma': float(g),
            'f_rev': float(f_rev)
        })
    print(f"Successfully extracted {len(dataset_params)} energy levels.")
    return dataset_params

# ******************************************************
#              3. SIGNAL GENERATOR MOTOR
# ******************************************************

def generate_and_save_dataset(energy, f_rev, gamma, state_name, enable_betatron, enable_momentum, add_noise):
    
    # Calculate exact physical slip factor (sigma)
    eta = 1.0 / gamma**2 - 1.0 / GAMMA_T**2
    
    T_rev = 1.0 / f_rev
    num_turns = int(TOTAL_TIME * f_rev)
    t_digital = np.arange(0, TOTAL_TIME, DT)
    sensor_signal = np.zeros_like(t_digital)

    # Set dynamic limits based on the state (Ablation study)
    dp_p = DP_P_MAX if enable_momentum else 0.0
    dA_A = DA_A_MAX if enable_betatron else 0.0
    
    # If no transverse oscillation, we need a baseline amplitude to see the longitudinal shifts
    A0 = 0.0 if enable_betatron else 1.0    

    T_new = 0.0
    for n in range(num_turns):
        t_ideal = (n + 1) * T_rev  
        
        # Longitudinal dynamics
        current_dp = dp_p * np.sin(2 * np.pi * F_W * t_ideal)
        T_new = T_new - T_rev * eta * current_dp
        absolute_time = t_ideal + T_new  
        
        # Transverse dynamics
        current_amplitude = A0 + dA_A * np.sin(2 * np.pi * F_A * f_rev * absolute_time)
        
        # Gaussian envelope
        idx_mask = (t_digital > absolute_time - 5 * SIGMA_T) & (t_digital < absolute_time + 5 * SIGMA_T)
        if np.any(idx_mask):
            sensor_signal[idx_mask] += current_amplitude * np.exp(-0.5 * ((t_digital[idx_mask] - absolute_time) / SIGMA_T)**2)

    # Add Noise if required
    if add_noise:
        max_amp = np.max(np.abs(sensor_signal))
        noise_level = 0.05 * max_amp if max_amp > 0 else 0.05 # 5% noise level
        sensor_signal += np.random.normal(0, noise_level, size=sensor_signal.shape)

    # Calculate FFT 
    n_fft = len(t_digital)
    freqs = np.fft.fftfreq(n_fft, d=DT)
    fft_values = np.fft.fft(sensor_signal)
    amplitude_spectrum = (np.abs(fft_values) / n_fft) * 2

    # Normalizing the spectrum
    max_spec = np.max(amplitude_spectrum)
    if max_spec > 0:
        amplitude_spectrum = amplitude_spectrum / max_spec

    # Filter DC to 5 MHz to reduce file size
    mask = (freqs >= 0) & (freqs <= 5e6) 
    freqs_masked = freqs[mask]
    amps_masked = amplitude_spectrum[mask]

    # Save to CSV
    filename = os.path.join(OUTPUT_DIR, f"Carbon_{int(energy)}MeV_{state_name}.csv")
    
    # Save as two columns: Frequency, Amplitude
    data_to_save = np.column_stack((freqs_masked, amps_masked))
    np.savetxt(filename, data_to_save, delimiter=",", header="Frequency_Hz,Normalized_Amplitude", comments="")
    
    # RAM Protection: Delete heavy arrays and force garbage collection
    del t_digital, sensor_signal, freqs, fft_values, amplitude_spectrum, freqs_masked, amps_masked, data_to_save
    gc.collect()

# ******************************************************
#                 4. MAIN EXECUTION LOOP
# ******************************************************
if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "carbon.txt")
    
    params = parse_proton_file(file_path)
    
    for p in params:
        energy = p['energy']
        gamma = p['gamma']
        f_rev = p['f_rev']
        
        print(f"\nProcessing Energy: {energy} MeV/u (Gamma: {gamma}, f_rev: {f_rev/1e6:.4f} MHz)")
        
        states = [
            {"name": "Transverse_Only_Clean", "beta": True,  "mom": False, "noise": False},
            {"name": "Momentum_Only_Clean",   "beta": False, "mom": True,  "noise": False},
            {"name": "Combined_Clean",        "beta": True,  "mom": True,  "noise": False},
            {"name": "Transverse_Only_Noise", "beta": True,  "mom": False, "noise": True},
            {"name": "Momentum_Only_Noise",   "beta": False, "mom": True,  "noise": True},
            {"name": "Combined_Noise",        "beta": True,  "mom": True,  "noise": True}
        ]
        
        for state in states:
            print(f"  -> Generating {state['name']}...", end="", flush=True)
            generate_and_save_dataset(
                energy=energy, 
                f_rev=f_rev, 
                gamma=gamma, 
                state_name=state['name'], 
                enable_betatron=state['beta'], 
                enable_momentum=state['mom'], 
                add_noise=state['noise']
            )
            print(" Done.")
            
    print("\nDataset generation completed successfully! Check the 'ML_Datasets' folder.")

# ******************************************************
#                 4. MAIN EXECUTION LOOP
# ******************************************************
if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "carbon.txt")
    
    params = parse_proton_file(file_path)
    
    for p in params:
        energy = p['energy']
        gamma = p['gamma']
        f_rev = p['f_rev']
        
        print(f"\nProcessing Energy: {energy} MeV/u (Gamma: {gamma}, f_rev: {f_rev/1e6:.4f} MHz)")
        
        states = [
            {"name": "Transverse_Only_Clean", "beta": True,  "mom": False, "noise": False},
            {"name": "Momentum_Only_Clean",   "beta": False, "mom": True,  "noise": False},
            {"name": "Combined_Clean",        "beta": True,  "mom": True,  "noise": False},
            {"name": "Transverse_Only_Noise", "beta": True,  "mom": False, "noise": True},
            {"name": "Momentum_Only_Noise",   "beta": False, "mom": True,  "noise": True},
            {"name": "Combined_Noise",        "beta": True,  "mom": True,  "noise": True}
        ]
        
        for state in states:
            print(f"  -> Generating {state['name']}...", end="", flush=True)
            generate_and_save_dataset(
                energy=energy, 
                f_rev=f_rev, 
                gamma=gamma, 
                state_name=state['name'], 
                enable_betatron=state['beta'], 
                enable_momentum=state['mom'], 
                add_noise=state['noise']
            )
            print(" Done.")
            
    print("\nDataset generation completed successfully! Check the 'ML_Datasets' folder.")