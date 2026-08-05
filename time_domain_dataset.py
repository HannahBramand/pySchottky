import numpy as np
import re
import os
import gc
import matplotlib.pyplot as plt
import argparse

# ******************************************************
#                   1. PARAMETERS
# ******************************************************

GAMMA_T = 1.971            
F_A = 5.0 / 3.0            
DP_P_MAX = 1e-3            
DA_A_MAX = 0.1             
F_W = 10e3                 
SIGMA_T = 70e-9            

TOTAL_TIME = 1e-3          
F_S = 200e6                
DT = 1.0 / F_S
OUTPUT_DIR = "ML_Datasets_TimeDomain"

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
#              3. SIGNAL GENERATOR MOTOR (Time Domain Only)
# ******************************************************

def generate_and_save_dataset(energy, f_rev, gamma, state_name, enable_betatron, enable_momentum, add_noise, lPlot=True):
    
    eta = 1.0 / gamma**2 - 1.0 / GAMMA_T**2
    
    T_rev = 1.0 / f_rev
    num_turns = int(TOTAL_TIME * f_rev)
    t_digital = np.arange(0, TOTAL_TIME, DT)
    sensor_signal = np.zeros_like(t_digital)

    dp_p = DP_P_MAX if enable_momentum else 0.0
    dA_A = DA_A_MAX if enable_betatron else 0.0
    A0 = 1.0   

    delta_t_history = []
    delta_p_p_history = []

    T_new = 0.0
    for n in range(num_turns):
        t_ideal = (n + 1) * T_rev  
        
        current_dp = dp_p * np.sin(2 * np.pi * F_W * t_ideal)
        T_new = T_new - T_rev * eta * current_dp
        absolute_time = t_ideal + T_new  
        
        delta_t_history.append(T_new)
        delta_p_p_history.append(current_dp / DP_P_MAX if DP_P_MAX > 0 else 0) 

        current_amplitude = A0 + dA_A * np.sin(2 * np.pi * F_A * f_rev * absolute_time)
        
        idx_mask = (t_digital > absolute_time - 5 * SIGMA_T) & (t_digital < absolute_time + 5 * SIGMA_T)
        if np.any(idx_mask):
            sensor_signal[idx_mask] += current_amplitude * np.exp(-0.5 * ((t_digital[idx_mask] - absolute_time) / SIGMA_T)**2)

    if add_noise:
        max_amp = np.max(np.abs(sensor_signal))
        noise_level = 0.05 * max_amp if max_amp > 0 else 0.05 
        sensor_signal += np.random.normal(0, noise_level, size=sensor_signal.shape)

    # ---------------------------------------------------------
    #                     PLOTTING ENGINE 
    # ---------------------------------------------------------
    if lPlot:
        plots_dir = os.path.join(OUTPUT_DIR, "Plots")
        os.makedirs(plots_dir, exist_ok=True)

        fig, axs = plt.subplots(2, 1, figsize=(12, 10)) 
        
        zoom_time = 5 * T_rev
        mask_time = t_digital <= zoom_time
        axs[0].plot(t_digital[mask_time] * 1e6, sensor_signal[mask_time], color='cyan', linewidth=1.5)
        axs[0].set_title(f'Raw Time Domain Signal: First 5 Turns ({state_name})', fontsize=14, fontweight='bold')
        axs[0].set_xlabel('Time (us)')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        axs[1].plot(np.array(delta_t_history) * 1e9, delta_p_p_history, color='cyan', marker='.', linestyle='', markersize=3, alpha=0.5)
        axs[1].set_title(f'Longitudinal Phase Space: {state_name}', fontsize=14, fontweight='bold')
        axs[1].set_xlabel('Time Deviation Δt (ns)')
        axs[1].set_ylabel('Normalized Momentum Deviation')
        axs[1].grid(True, linestyle='--', alpha=0.6)
      
        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, f"carbon_{int(energy)}MeV_{state_name}_TimeDomain.png")
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)  

    # ---------------------------------------------------------
    #                     SAVING DATASET (Time Domain)
    # ---------------------------------------------------------
    filename = os.path.join(OUTPUT_DIR, f"carbon{int(energy)}MeV_{state_name}_Time.csv")
    data_to_save = np.column_stack((t_digital, sensor_signal))
    
    # Save purely Time vs Amplitude
    np.savetxt(filename, data_to_save, delimiter=",", header="Time_s,Amplitude", comments="")
    
    # Clean up memory
    del t_digital, sensor_signal, data_to_save
    del delta_t_history, delta_p_p_history
    gc.collect()


# ******************************************************
#                 4. MAIN EXECUTION LOOP 
# ******************************************************
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Raw Time-Domain Schottky Datasets.")
    
    parser.add_argument('--no-plot', action='store_false', dest='lPlot',
                        help='Plotting is enabled by default.')
    parser.set_defaults(lPlot= True)

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
            
    print("\nDataset generation completed! Check the 'ML_Datasets_TimeDomain' folder.")