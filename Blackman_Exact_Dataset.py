import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import re
import gc

def parse_carbon_file(filepath):
    print(f"Reading physics config from {filepath}...")
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
    return dataset_params

def calculate_local_snr(spectrum_amps, peak_index, window_size=100):
    peak_amp = spectrum_amps[peak_index]
    
    start = max(0, peak_index - window_size)
    end = min(len(spectrum_amps), peak_index + window_size)
    
    floor_region = np.concatenate((spectrum_amps[start:max(0, peak_index-10)], 
                                   spectrum_amps[min(len(spectrum_amps), peak_index+10):end]))
    
    local_floor = np.median(floor_region) if len(floor_region) > 0 else 1e-12
    if local_floor > 0:
        return 20 * np.log10(peak_amp / local_floor)
    return 0.0

def get_exact_blackman_window(N):
    if N == 1:
        return np.array([1.0])
        
    a0 = 7938 / 18608
    a1 = 9240 / 18608
    a2 = 1430 / 18608
    
    n_idx = np.arange(N)
    window = a0 - a1 * np.cos(2 * np.pi * n_idx / (N - 1)) + a2 * np.cos(4 * np.pi * n_idx / (N - 1))
    return window

def process_and_save_exact_blackman(csv_filepath, true_f_rev, output_name, plot_dir, data_dir):
    print(f"  -> Processing: {output_name}...")
    
    df = pd.read_csv(csv_filepath)
    
    if 'Frequency_Hz' in df.columns:
        print("     [Error] Frequency Domain data detected! Skipping.")
        return
        
    if 'Time_s' not in df.columns or 'Amplitude' not in df.columns:
        time_s = df.iloc[:, 0].values
        signal = df.iloc[:, 1].values
    else:
        time_s = df['Time_s'].values
        signal = df['Amplitude'].values
        
    N = len(signal)
    dt = time_s[1] - time_s[0] if len(time_s) > 1 else 1e-8
    
    raw_fft = np.fft.rfft(signal)
    raw_freqs = np.fft.rfftfreq(N, d=dt)
    raw_amps = np.abs(raw_fft) / N
    mask_raw = raw_freqs > 50e3
    plot_raw_freqs = raw_freqs[mask_raw]
    plot_raw_amps = raw_amps[mask_raw]
    
    window = get_exact_blackman_window(N)
    windowed_signal = signal * window
    
    win_fft = np.fft.rfft(windowed_signal)
    win_freqs = np.fft.rfftfreq(N, d=dt)
    win_amps = np.abs(win_fft) / N
    
    mask_win = win_freqs > 50e3
    plot_win_freqs = win_freqs[mask_win]
    plot_win_amps = win_amps[mask_win]

    if len(plot_win_amps) == 0:
        print("     [Error] No valid frequencies found.")
        return

    dataset_filename = os.path.join(data_dir, f"{output_name}_ExactBlackman.csv")
    data_to_save = np.column_stack((plot_win_freqs, plot_win_amps))
    np.savetxt(dataset_filename, data_to_save, delimiter=",", header="Frequency_Hz,Normalized_Amplitude", comments="")

    peaks, _ = find_peaks(plot_win_amps, height=np.max(plot_win_amps) * 0.1, distance=50)
    
    detected_f_rev = 0
    peak_amp = 0
    if len(peaks) > 0:
        main_peak_idx = peaks[np.argmax(plot_win_amps[peaks])]
        detected_f_rev = plot_win_freqs[main_peak_idx]
        peak_amp = plot_win_amps[main_peak_idx]
        snr_db = calculate_local_snr(plot_win_amps, main_peak_idx)
        
        error_hz = abs(detected_f_rev - true_f_rev)
        error_pct = (error_hz / true_f_rev) * 100
        
        print(f"     [+] f_rev: {detected_f_rev:.2f} Hz | Error: {error_pct:.4f}% | SNR: {snr_db:.2f} dB")
    else:
        print("     [-] No prominent peaks detected.")

    # =========================================================================
    # PLOT 1: STANDARD FULL VIEW
    # =========================================================================
    fig_std, axs_std = plt.subplots(3, 1, figsize=(12, 14))
    
    zoom_points = min(N, int(N * 0.05)) if N > 2000 else N
    axs_std[0].plot(time_s[:zoom_points] * 1e6, signal[:zoom_points], color='cyan', linewidth=1.2)
    axs_std[0].set_title(f"1. Raw Time Domain (First 5%) - {output_name}", fontsize=13, fontweight='bold')
    axs_std[0].set_xlabel("Time [us]")
    axs_std[0].set_ylabel("Amplitude")
    axs_std[0].grid(True, linestyle='--', alpha=0.6)
    
    axs_std[1].plot(plot_raw_freqs / 1e6, plot_raw_amps, color='orange', linewidth=1.2)
    axs_std[1].set_title("2. Raw Frequency Spectrum (No Window)", fontsize=13, fontweight='bold')
    axs_std[1].set_xlabel("Frequency [MHz]")
    axs_std[1].set_ylabel("Magnitude")
    axs_std[1].grid(True, linestyle='--', alpha=0.6)
    
    axs_std[2].plot(plot_win_freqs / 1e6, plot_win_amps, color='#2ca02c', linewidth=1.2)
    if detected_f_rev > 0:
        axs_std[2].plot(detected_f_rev / 1e6, peak_amp, "ro", label=f"Detected f_rev: {detected_f_rev/1e6:.4f} MHz")
        axs_std[2].legend()
    axs_std[2].set_title("3. Windowed Spectrum (Exact Blackman)", fontsize=13, fontweight='bold')
    axs_std[2].set_xlabel("Frequency [MHz]")
    axs_std[2].set_ylabel("Magnitude")
    axs_std[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_filename_std = os.path.join(plot_dir, f"{output_name}_ExactBlackman.png")
    fig_std.savefig(plot_filename_std, dpi=150)
    plt.close(fig_std) 

    # =========================================================================
    # PLOT 2: ZOOMED VIEW
    # =========================================================================
    fig_zoom, axs_zoom = plt.subplots(3, 1, figsize=(12, 14))
    
    T_rev = 1.0 / true_f_rev
    zoom_time_limit = 3 * T_rev 
    mask_time_zoom = time_s <= zoom_time_limit
    
    axs_zoom[0].plot(time_s[mask_time_zoom] * 1e6, signal[mask_time_zoom], color='cyan', linewidth=1.5, marker='.')
    axs_zoom[0].set_title(f"1. ZOOMED Time Domain (First 3 Turns) - {output_name}", fontsize=13, fontweight='bold')
    axs_zoom[0].set_xlabel("Time [us]")
    axs_zoom[0].set_ylabel("Amplitude")
    axs_zoom[0].grid(True, linestyle='--', alpha=0.6)
    
    center_freq = detected_f_rev if detected_f_rev > 0 else true_f_rev
    span_hz = 50e3  # +/- 50 kHz
    
    mask_raw_zoom = (plot_raw_freqs >= center_freq - span_hz) & (plot_raw_freqs <= center_freq + span_hz)
    axs_zoom[1].plot(plot_raw_freqs[mask_raw_zoom] / 1e6, plot_raw_amps[mask_raw_zoom], color='orange', linewidth=1.5, marker='.')
    axs_zoom[1].set_title(f"2. ZOOMED Raw Frequency Spectrum (Centered at {center_freq/1e6:.4f} MHz)", fontsize=13, fontweight='bold')
    axs_zoom[1].set_xlabel("Frequency [MHz]")
    axs_zoom[1].set_ylabel("Magnitude")
    axs_zoom[1].grid(True, linestyle='--', alpha=0.6)
    
    mask_win_zoom = (plot_win_freqs >= center_freq - span_hz) & (plot_win_freqs <= center_freq + span_hz)
    axs_zoom[2].plot(plot_win_freqs[mask_win_zoom] / 1e6, plot_win_amps[mask_win_zoom], color='#2ca02c', linewidth=1.5, marker='.')
    if detected_f_rev > 0:
        axs_zoom[2].plot(detected_f_rev / 1e6, peak_amp, "ro", markersize=8, label=f"Peak: {detected_f_rev/1e6:.4f} MHz")
        axs_zoom[2].legend()
    axs_zoom[2].set_title("3. ZOOMED Windowed Spectrum (Exact Blackman)", fontsize=13, fontweight='bold')
    axs_zoom[2].set_xlabel("Frequency [MHz]")
    axs_zoom[2].set_ylabel("Magnitude")
    axs_zoom[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_filename_zoom = os.path.join(plot_dir, f"{output_name}_ExactBlackman_Zoomed.png")
    fig_zoom.savefig(plot_filename_zoom, dpi=150)
    plt.close(fig_zoom)

if __name__ == "__main__":
    print("\n==================================================")
    print("   AUTOMATED BATCH WINDOWING (EXACT BLACKMAN)     ")
    print("==================================================\n")
    
    # !!! IMPORTANT: PUT YOUR ABSOLUTE PATH HERE !!!
    INPUT_DIR = "/home/hannah/Hannah/all files/UniPv/Thesis/Table/new/C-ion/time_domain/without-noise/trans+longi ( BOTH )" 
    
    OUTPUT_DATA_DIR = "Windowed_Datasets"
    OUTPUT_PLOT_DIR = "Windowing_Plots"
    
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_folder, "carbon.txt")
    
    try:
        params = parse_carbon_file(config_file)
    except FileNotFoundError:
        print(f"[Fatal Error] Could not find '{config_file}'. Please ensure it is in the same folder.")
        exit()

    states = [
        "Transverse_Only_Clean", "Momentum_Only_Clean", "Combined_Clean",
        "Transverse_Only_Noise", "Momentum_Only_Noise", "Combined_Noise"
    ]
    
    total_files_processed = 0
    
    for p in params:
        energy = int(p['energy'])
        true_f_rev = p['f_rev']
        
        print(f"\n--- Analyzing Energy Level: {energy} MeV/u ---")
        
        for state in states:
            file_name = f"carbon{energy}MeV_{state}_Time.csv"
            input_file = os.path.join(INPUT_DIR, file_name)
            
            output_name = f"carbon{energy}MeV_{state}"
            
            if os.path.exists(input_file):
                process_and_save_exact_blackman(input_file, true_f_rev, output_name, OUTPUT_PLOT_DIR, OUTPUT_DATA_DIR)
                total_files_processed += 1
                gc.collect() 
            else:
                print(f"  -> Skipping: {file_name} (File not found)")
                
    print(f"\n==================================================")
    print(f" Process completed! {total_files_processed} files successfully windowed.")
    print("==================================================")