import numpy as np
import re
import os
import gc
import matplotlib.pyplot as plt
import argparse

# ******************************************************
#                   1. PARAMETERS
# ******************************************************

GAMMA_T = 1.971            # Transition Gamma for CNAO
F_A = 5.0 / 3.0            # Fractional Betatron Tune 
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

def parse_carbon_file(filepath):
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
#              3. SIGNAL GENERATOR MOTOR (Updated with Plotting)
# ******************************************************

def generate_and_save_dataset(energy, f_rev, gamma, state_name, enable_betatron, enable_momentum, add_noise, lPlot=True):
    
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

    delta_t_history = []
    delta_p_p_history = []
    # -------------------------------------------------------

    T_new = 0.0
    for n in range(num_turns):
        t_ideal = (n + 1) * T_rev  
        
        # Longitudinal dynamics
        current_dp = dp_p * np.sin(2 * np.pi * F_W * t_ideal)
        T_new = T_new - T_rev * eta * current_dp
        absolute_time = t_ideal + T_new  
        
        delta_t_history.append(T_new)
        delta_p_p_history.append(current_dp / DP_P_MAX if DP_P_MAX > 0 else 0) 
        # ----------------------------------------

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

    # ---------------------------------------------------------
    #                     PLOTTING ENGINE (Updated for 4 plots and Control)
    # ---------------------------------------------------------
    if lPlot:
        plots_dir = os.path.join(OUTPUT_DIR, "Plots")
        os.makedirs(plots_dir, exist_ok=True)

        fig, axs = plt.subplots(4, 1, figsize=(12, 20)) 
        
        zoom_time = 5 * T_rev
        mask_time = t_digital <= zoom_time
        axs[0].plot(t_digital[mask_time] * 1e6, sensor_signal[mask_time], color='magenta', linewidth=1.5)
        axs[0].set_title(f'Time Domain: First 5 Turns ({state_name})', fontsize=14, fontweight='bold')
        axs[0].set_xlabel('Time (us)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        freqs_mhz = freqs_masked / 1e6
        axs[1].plot(freqs_mhz, amps_masked, color='magenta', linewidth=1)
        axs[1].set_title('Frequency Domain: Full Spectrum (0 to 5 MHz)', fontsize=14, fontweight='bold')
        axs[1].set_xlabel('Frequency (MHz)')
        axs[1].set_ylabel('Normalized Amplitude')
        axs[1].set_xlim(0, 5)
        axs[1].grid(True, linestyle='--', alpha=0.6)

        mask_no_dc = freqs_mhz > 0.1 
        if np.any(mask_no_dc): 
            max_peak_freq = freqs_mhz[mask_no_dc][np.argmax(amps_masked[mask_no_dc])]
            span_mhz = 0.05  
            mask_zoom = (freqs_mhz >= max_peak_freq - span_mhz) & (freqs_mhz <= max_peak_freq + span_mhz)
            
            axs[2].plot(freqs_mhz[mask_zoom], amps_masked[mask_zoom], color='magenta', linewidth=1.5)
            axs[2].set_title(f'Smart Zoomed FFT: Centered around {max_peak_freq:.3f} MHz', fontsize=14, fontweight='bold')
            axs[2].set_xlabel('Frequency (MHz)')
            axs[2].set_ylabel('Amplitude')
            axs[2].grid(True, linestyle='--', alpha=0.6)
        else:
            axs[2].set_title("Smart Zoomed FFT: No Significant Peak Found")

        axs[3].plot(np.array(delta_t_history) * 1e9, delta_p_p_history, color='magenta', marker='.', linestyle='', markersize=3, alpha=0.5)
        axs[3].set_title(f'Longitudinal Phase Space: {state_name}', fontsize=14, fontweight='bold')
        axs[3].set_xlabel('Time Deviation Δt (ns)')
        axs[3].set_ylabel('Normalized Momentum Deviation (Δp/p) / (Δp/p)_max')
        axs[3].grid(True, linestyle='--', alpha=0.6)
        # ----------------------------------------------
      
        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, f"Carbon_{int(energy)}MeV_{state_name}.png")
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)  
    # ---------------------------------------------------------

    filename = os.path.join(OUTPUT_DIR, f"Carbon{int(energy)}MeV_{state_name}.csv")
    data_to_save = np.column_stack((freqs_masked, amps_masked))
    np.savetxt(filename, data_to_save, delimiter=",", header="Frequency_Hz,Normalized_Amplitude", comments="")
    
    del t_digital, sensor_signal, freqs, fft_values, amplitude_spectrum, freqs_masked, amps_masked, data_to_save
    del delta_t_history, delta_p_p_history
    gc.collect()


# ******************************************************
#                 4. MAIN EXECUTION LOOP (Updated for Command Line Control)
# ******************************************************
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Schottky Signal Datasets.")
    
    parser.add_argument('--no-plot', action='store_false', dest='lPlot',
                        help='Plotting is enabled by default.')
    parser.set_defaults(lPlot= True)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 ######## if you want plot turn the upper on to TRUE... :)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    args = parser.parse_args()
    
    print(f"--- Dataset Generation Started (Plotting is {'ENABLED' if args.lPlot else 'DISABLED'}) ---")

    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "carbon.txt")
    
    params = parse_carbon_file(file_path)
    
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
                add_noise=state['noise'],
                lPlot=args.lPlot 
            )
            print(" Done.")
            
    print("\nDataset generation completed successfully! Check the 'ML_Datasets' folder.")