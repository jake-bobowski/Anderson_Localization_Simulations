# Jake S. Bobowski
# University of British Columbia
# Physics Department
# August 29, 2025
# Point Particle Diffusion/Anderson Localization Simulation

def generate_energy_frames(v_map_accum, t_ds, V_k, f, f0, sigma_f, f0_factor, mfp, ell, disorder_scale, frame_dir, time_range=(0, 100e-9), n_frames=501):
    """
    Generate frames showing:
    - ⟨|v_k(t)|²⟩ vs k (top panel)
    - |V_k(f)|² vs f at peak energy k (bottom panel)

    Parameters:
        v_map_accum: shape [n_time, N], ensemble-averaged |v_k(t)|^2
        t_ds: downsampled time array (same length as v_map_accum)
        V_k: dict of frequency-domain voltage V_k[f] per segment (not squared)
        f: frequency array (Hz), same for all k
        f0: center frequency of input pulse (Hz)
        sigma_f: standard deviation of input pulse (Hz)
        f0_factor: scaling factor for f0
        mfp: mean free path (m)
        ell: system size (m)
        disorder_scale: disorder strength (dimensionless)
        frame_dir: folder to save PNG frames
        time_range: tuple (start_time, stop_time) in seconds
        n_frames: number of animation frames to generate
    """
    
    # Number of TL segments
    N = v_map_accum.shape[1]
    k_vals = np.arange(N)
    
    # Restrict to indices within time_range
    time_mask = (t_ds >= time_range[0]) & (t_ds <= time_range[1])
    t_ds_trimmed = t_ds[time_mask]
    v_map_accum_trimmed = v_map_accum[time_mask, :]  # ← fix

    # Interpolate to n_frames between time_range
    t_frame_vals = np.linspace(time_range[0], time_range[1], n_frames)

    # Wave speed
    v0 = mfp * f0 / f0_factor 
    for i, t_target in tqdm(enumerate(t_frame_vals), desc="Forming frame", ncols=80):
        # Interpolate v_k_t to exact frame time
        v_k_t = np.empty(N)
        for k in range(N):
            v_k_t[k] = np.interp(t_target, t_ds_trimmed, v_map_accum_trimmed[:, k])

        # Suppress spectrum outside the TL transit window
        pulse_entry = 1 / sigma_f
        transit_time = ell / v0
        pulse_exit = pulse_entry + transit_time
        return_time = pulse_exit + transit_time

        if t_target < pulse_entry:
            k_peak = None
            V_k_mag2 = np.zeros_like(f)

        elif pulse_entry <= t_target < pulse_exit:
            # Pulse traveling forward through TL
            frac = (t_target - pulse_entry) / transit_time
            k_peak = int(np.clip(round(frac * N), 0, N - 1))
            V_k_f = V_k[k_peak]
            V_k_mag2 = np.abs(V_k_f) ** 2

        elif pulse_exit <= t_target < return_time:
            # Pulse returning back toward source
            frac = (t_target - pulse_exit) / transit_time
            k_peak = int(np.clip((1 - frac) * N, 0, N - 1))
            V_k_f = V_k[k_peak]
            V_k_mag2 = np.abs(V_k_f) ** 2

        else:
            # After second pass — energy presumed at k = 0
            k_peak = 0
            V_k_f = V_k[k_peak]
            V_k_mag2 = np.abs(V_k_f) ** 2

        # --- Trim frequency axis to relevant window ---
        n_sigma = 2.5  # Or 2.0 for tighter window
        f_min = f0 - n_sigma * sigma_f
        f_max = f0 + n_sigma * sigma_f

        # Mask the frequency and spectrum arrays
        freq_mask = (f >= f_min) & (f <= f_max)
        f_plot = f[freq_mask] * 1e-9  # Convert to GHz
        V_k_plot = V_k_mag2[freq_mask]
        if np.max(V_k_plot) > 0:
            V_k_plot = V_k_plot / np.max(V_k_plot)

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # ⟨|v_k(t)|²⟩ vs k
        ax1.plot(k_vals, v_k_t, 'b-')
        ax1.set_xlim(0, N - 1)
        if disorder_scale == 0:
            ax2.set_yscale('linear')
            ax1.set_ylim(0, 1)
        else:
            ax1.set_yscale('log')
            ax1.set_ylim(2e-4, 0.3)    
        ax1.set_ylabel(r'$\langle |v_k(t)|^2 - \overline{|v_k(t)|^2}_{\mathrm{bg}} \rangle$ (V$^2$)')
        ax1.set_title(f'Time = {t_target * 1e9:.1f} ns')
        ax1.grid(True)

        # |V_k(f)|² vs frequency at k_peak
        ax2.plot(f_plot, V_k_plot, 'r-')
        if disorder_scale == 0:
            ax2.set_yscale('linear')
            ax2.set_ylim(0, 1.05)
        else:
            ax2.set_yscale('log')
            ax2.set_ylim(1e-8, 1.1)
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel(r'$|V_k(f)|^2$ (V$^2$/Hz)')
        if k_peak is not None:
            ax2.set_title(f'Spectrum at k = {k_peak}')
        else:
            ax2.set_title('Spectrum (no energy in TL)')
        ax2.grid(True)

        # Save the time- and freq-domain data to text files.
        # Can be used to generate vector graphics using external tools (pgfplots).
        # Commented out if not needed.
        np.savetxt(os.path.join(frame_dir, f"frame_{i:03d}_t{t_target*1e9:.1f}ns_time.txt"), np.column_stack((k_vals, v_k_t)),
                  header="k\tnormalizedEnergy", fmt="%.10e %.10e")
        np.savetxt(os.path.join(frame_dir, f"frame_{i:03d}_t{t_target*1e9:.1f}ns_freq.txt"), np.column_stack((f_plot, V_k_plot)),
                  header="frequency\tspectrum", fmt="%.10e %.10e")

        plt.tight_layout()
        plt.savefig(os.path.join(frame_dir, f"frame_{i:03d}_t{t_target*1e9:.1f}ns.png"), dpi=600)
        plt.close()


def transfer_matrix(Z0_k, v0_k, ell_k, omega):

    """
    Return the 2×2 transfer matrix T_k(ω) for a TL segment.
    omega can be a scalar or numpy array.
    """
    
    beta_k = omega / v0_k
    cos_bl = np.cos(beta_k * ell_k)
    sin_bl = np.sin(beta_k * ell_k)

    T_k = np.array( [ [ cos_bl, 1j * Z0_k * sin_bl ], [ ( 1j/Z0_k ) * sin_bl, cos_bl ] ] )
    return T_k

def TL_AL(N = 500, n_freq = 2**20, mfp = 0.15, disorder_scale = 0.5, disorder_onset = 0.2,  f0_factor = 2.0, sigma_divisor = 20):

    # --- Constants ---
    Zs = 50         # ohms
    c = 3e8         # m/s
    v0 = 0.7 * c    # wave speed
    Z0 = 50         # ohms
    V0 = 1          # input voltage amplitude
    
    # --- Frequency grid ---
    f0 = f0_factor * v0 / mfp
    sigma_f = f0 / sigma_divisor
    t0 = 1 / sigma_f
    
    f_max = 10 * f0
    df = 2 * f_max / n_freq
    f = np.linspace(-f_max, f_max - df, n_freq)
    t = np.fft.fftshift(np.fft.fftfreq(n_freq, d=df))
    omega = 2 * np.pi * f
    
    # Construct full-spectrum Gaussian
    Vs = V0 / (sigma_f * np.sqrt(2* np.pi)) * np.exp( -0.5 * ( ( f - f0 ) / sigma_f)**2 ) * np.exp( -1j * omega * t0 )
    
    # --- Transmission line disorder ---
    muC = 1 / (Z0 * v0)
    muL = Z0 / v0
    sigmaC = muC * disorder_scale
    sigmaL = muL * disorder_scale
    
    # Select randomized C, L, and ell
    L, C = [], []
    disorder_onset = disorder_onset * N
    for k in range(N):
        adiabatic = (1 - np.exp(-k / disorder_onset))**2
        L.append(np.random.normal(muL, sigmaL * adiabatic))
        C.append(np.random.normal(muC, sigmaC * adiabatic))
    
    L = np.clip(L, 0.1 * muL, 4.0 * muL)
    C = np.clip(C, 0.1 * muC, 4.0 * muC)
    L = np.array(L)
    C = np.array(C)

    if disorder_scale == 0:
        ell_k = np.ones(N) * mfp  # uniform length for simplicity
    else:
        ell_k = np.random.exponential(scale=mfp, size=N)
        ell_k = np.clip(ell_k, 0.1 * mfp, 4.0 * mfp)
    ell = sum(ell_k)

    v0k = np.sqrt(1 / (L * C))
    Z0k = np.sqrt(L / C)

    Tk_dict = {}

    # Calculate T_Net which can be used to find Zin_Net
    T_Net = np.tile(np.eye(2)[None, :, :], (n_freq, 1, 1))  # shape: (n_freq, 2, 2)
    for k in range(N - 1, -1, -1):
        Tk = transfer_matrix(Z0k[k], v0k[k], ell_k[k], omega)
        Tk = np.transpose(Tk, (2, 0, 1))  # Now (n_freq, 2, 2)
        
        formatted = f"{k:03d}"
        Tk_dict['Tk' + formatted] = Tk
        Tk_dict['Tk_inv' + formatted] = np.linalg.inv(Tk)
        T_Net = Tk @ T_Net

    # Compute Zin_Net at each frequency
    T21 = T_Net[:, 1, 0]
    T11 = T_Net[:, 0, 0]
    
    # Threshold for numerical zero
    eps = 1e-12
    
    # Robust impedance calculation
    Zin_Net_k = np.empty_like(T11)
    bad = np.abs(T21) < eps
    good = ~bad
    
    # Compute where valid
    Zin_Net_k[good] = T11[good] / T21[good]
    
    # Set invalid points to NaN or large value (e.g., open circuit)
    large_imp = 1e12
    Zin_Net_k[bad] = 1j * np.sign(T11[bad]) * np.sign(T21[bad]) * large_imp

    VI_dict = {}

    # Compute voltage divider factor at each frequency
    VI_prefactor = Vs / (Zin_Net_k + Zs)
    
    # Construct 2×1 "source vector" T_k(ω) for each frequency — proper shape: (n_freq, 2, 1)
    VI_source_vec = np.zeros((n_freq, 2, 1), dtype=complex)
    VI_source_vec[:, 0, 0] = Zin_Net_k
    VI_source_vec[:, 1, 0] = 1.0
    
    VI_dict['VI000'] = VI_prefactor[:, None, None] * VI_source_vec  # shape (n_freq, 2, 1)
    
    for k in range(1, N + 1, 1):
        formatted_k = f"{k:03d}"
        formatted_k_minus = f"{k - 1:03d}"
        VI_dict['VI' + formatted_k] = Tk_dict['Tk_inv' + formatted_k_minus] @  VI_dict['VI' + formatted_k_minus]  

    V_k = {}
    I_k = {}
    v_k = {}
    i_k = {}
    for k in range(N + 1):
        VI_k = VI_dict[f'VI{k:03d}']
        V_k[k] = VI_k[:, 0, 0]  # shape: (n_freq,)
        I_k[k] = VI_k[:, 1, 0]   
        v_k[k] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(V_k[k]))) * n_freq * df
        i_k[k] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(I_k[k]))) * n_freq * df

    # rms V_k at f0
    idx_f0 = np.argmin(np.abs(f - f0))
    V_k_f0 = np.abs([V_k[k][idx_f0] for k in range(N + 1)])

    return V_k, V_k_f0, f, f0, sigma_f, v_k, t, ell


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering simulation")
    parser.add_argument("--N_segments", type=int, default=500, help="Number of segments")
    parser.add_argument("--n_freq", type=int, default=2**20, help="Number of frequencies")
    parser.add_argument("--N_iter", type=int, default=5000, help="Number of iterations")
    parser.add_argument("--mfp", type=float, default=0.15, help="Mean free path")
    parser.add_argument("--disorder_scale", type=float, default=0.5, help="TL disorder scale")
    parser.add_argument("--disorder_onset", type=float, default=0.2, help="TL disorder onset")
    parser.add_argument("--output_dir", type=str, default="heatmaps_avg", help="Output directory for results")
    parser.add_argument("--f0_factor", type=float, default=2.0, help="Factor for f0 adjustment")
    parser.add_argument("--sigma_divisor", type=float, default=20.0, help="Divisor for sigma_f calculation")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving heat maps")


    args = parser.parse_args()
    N_seg = args.N_segments
    n_freq = args.n_freq
    N_iter = args.N_iter
    mfp = args.mfp
    disorder_scale = args.disorder_scale
    disorder_onset = args.disorder_onset
    output_dir = args.output_dir
    f0_factor = args.f0_factor
    sigma_divisor = args.sigma_divisor
    save_interval = args.save_interval


    print(f"Run started at {datetime.datetime.now().isoformat()}")

    # -------------------------------
    # Parameters
    # -------------------------------
    step = 100                     # Downsampling factor in time
    time_unit = 1e6                # Convert time to µs
    position_unit = 1.0            # Use mfp if desired
    log_eps = 1e-20                # Floor for log scale to avoid log(0)
    
    os.makedirs(output_dir, exist_ok=True)

    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    temporal_dir = os.path.join(output_dir, "temporal")
    os.makedirs(temporal_dir, exist_ok=True)

    spectral_dir = os.path.join(output_dir, "spectral")
    os.makedirs(spectral_dir, exist_ok=True)

    # Log parameter set
    with open(os.path.join(output_dir, "parameters.txt"), "w") as paramfile:
        paramfile.write(f"Run started at {datetime.datetime.now().isoformat()}\n")
        paramfile.write(f"N = {N_seg}\n")
        paramfile.write(f"n_freq = {n_freq}\n")
        paramfile.write(f"N_iter = {N_iter}\n")
        paramfile.write(f"mfp = {mfp}\n")
        paramfile.write(f"disorder_scale = {disorder_scale}\n")
        paramfile.write(f"disorder_onset = {disorder_onset}\n")
        paramfile.write(f"N_iter = {N_iter}\n")
        paramfile.write(f"f0_factor = {f0_factor}\n")
        paramfile.write(f"sigma_divisor = {sigma_divisor}\n")
        paramfile.write(f"save_interval = {save_interval}\n")

    # -------------------------------
    # Initial Call to determine sizes
    # -------------------------------
    V_k, V_k_f0, f, f0, sigma_f, v_k, t, ell = TL_AL(N = N_seg, n_freq = n_freq, mfp = mfp, disorder_scale = disorder_scale, disorder_onset = disorder_onset, f0_factor = f0_factor, sigma_divisor = sigma_divisor)
    sorted_keys = sorted(v_k.keys())
    N = len(sorted_keys)
    n_freq = len(next(iter(v_k.values())))

    t0 = 1/sigma_f
    v0 = f0 * mfp / f0_factor  # wave speed
    ToF = t0 + 2 * ell/v0 # time of flight

    # FFT setup
    t_full = t
    t_mask = t > 0
    t = t[t_mask] # Trim negative times
    t_ds = t[::step]
    extent = [0, N * position_unit, t_ds[0] * time_unit, t_ds[-1] * time_unit]

    # -------------------------------
    # Initialize accumulators
    # -------------------------------
    V_k_f0_avg = np.zeros_like(V_k_f0)
    v_map_accum = np.zeros((len(t), N), dtype=np.float64)
    V_k_accum = {k: np.zeros_like(V_k[k], dtype=complex) for k in V_k}

    # NEW: Time-binned accumulators
    max_us = int(np.floor(t[-1] * 1e6))  # Final full µs bin
    bin_edges = np.arange(0, (max_us + 1) * 1e-6, 1e-6)  # in seconds
    bin_indices = np.digitize(t, bin_edges) - 1  # bin for each t
    v_k_binned_ens = {i: np.zeros(N) for i in range(max_us + 1)}

    # -------------------------------
    # Ensemble Loop
    # -------------------------------
    for i in tqdm(range(1, N_iter + 1), desc="Simulating", ncols=80):

        # --- Run Simulation ---
        V_k_new, V_k_f0_new, _, _, _, v_k_new, _, _ = TL_AL(N = N_seg, n_freq = n_freq, mfp = mfp, disorder_scale = disorder_scale, disorder_onset = disorder_onset, f0_factor = f0_factor, sigma_divisor = sigma_divisor)
        
        V_k_f0_new = V_k_f0_new**2
        V_k_f0_avg += (V_k_f0_new - V_k_f0_avg) / i

        # Compute 2D array of |V_k(f)|^2 for this realization
        V_k_mag2_array = np.array([np.abs(V_k_new[k])**2 for k in sorted_keys])

        # Frequency axis in GHz
        f_GHz = f * 1e-9
        f_min = f0 - 2.5 * sigma_f
        f_max = f0 + 2.5 * sigma_f
        freq_mask = (f >= f_min) & (f <= f_max)

        # --- Initialize spectral energy map accumulator (NEW) ---
        if i == 1:
            V_k_mag2_avg = np.zeros_like(V_k_mag2_array)

        # --- Accumulate ensemble-averaged |V_k(f)|^2 map ---
        V_k_mag2_avg += (V_k_mag2_array - V_k_mag2_avg) / i

        for k in V_k:
            V_k_accum[k] += (V_k_new[k] - V_k_accum[k]) / i

        # --- Save frequency-domain energy ---
        k_vals = np.arange(N)
        np.savetxt(os.path.join(output_dir, "V_k_f0_energy.txt"), np.column_stack((k_vals, V_k_f0_avg)),
                  header="k\t<V_k(f0)^2> (V^2)", fmt="%-8d %.10e")

        # --- Build v_map: |v_k(t)|^2 ---
        v_map = np.array([np.abs(v_k_new[k])**2 for k in sorted_keys]).T
        v_map = v_map[t_mask, :]  # Remove t < 0

        if disorder_scale != 0:
            # Apply background subtraction for disordered case
            # Compute background as average over early times: t in [0, T0/2]
            T0 = sigma_divisor / f0
            half_T0 = T0 / 2
            early_mask = t_full[t_mask] < half_T0

            v_k_bg = np.mean(v_map[early_mask, :], axis=0)  # shape: (N,)
            v_map_corr = np.maximum(v_map - v_k_bg[None, :], 1e-12)
        else:
            # No correction for homogeneous case
            v_map_corr = v_map

        v_map_accum += (v_map_corr - v_map_accum) / i


        # --- NEW: Time-binned averages ---
        for b in range(max_us + 1):
            bin_mask = bin_indices == b
            if np.any(bin_mask):
                v_k_bin_avg = np.mean(v_map[bin_mask, :], axis=0)
                v_k_binned_ens[b] += (v_k_bin_avg - v_k_binned_ens[b]) / i

                np.savetxt(os.path.join(output_dir, f"v_k_time_window_{b:03d}.txt"),
                          np.column_stack((k_vals, v_k_binned_ens[b])),
                          header=f"k\t<|v_k(t)|^2> over {b}–{b+1} µs",
                          fmt="%-8d %.10e")

        # --- Downsample and log scale ---
        v_map_ds = v_map_accum[::step, :]
        log_map = np.log10(v_map_ds + log_eps)

        # --- Total energy vs time ---
        E_total = np.sum(v_map_ds, axis=1)
        E_total_scaled = E_total / np.max(E_total)
        np.savetxt(os.path.join(output_dir, "E_total_vs_time.txt"), np.column_stack((t_ds * time_unit, E_total_scaled)),
                  header="Time (µs)\tNormalized Energy", fmt="%.3f %.10e")

         # --- Save plots at selected intervals ---
        save_this_iter = (i == 1) or (i == N_iter) or (i % save_interval == 0)

        if save_this_iter:
            label = "final" if i == N_iter else f"{i:04d}"

            # Trim and mask
            V_k_mag2_trimmed = V_k_mag2_avg[:, freq_mask]  # Use average
            f_GHz_trimmed = f_GHz[freq_mask]

            # Save image
            plt.figure(figsize=(10, 6))
            if disorder_scale == 0:
                plt.imshow(np.log10(V_k_mag2_trimmed), extent=[f_GHz_trimmed[0], f_GHz_trimmed[-1], 0, N],
                        aspect='auto', origin='lower', cmap='hot', vmin=-25, vmax=-15)
            else:
                plt.imshow(np.log10(V_k_mag2_trimmed), extent=[f_GHz_trimmed[0], f_GHz_trimmed[-1], 0, N],
                        aspect='auto', origin='lower', cmap='hot')

            plt.colorbar(label=r'$\log_{10}(|V_k(f)|^2)$ (V$^2$/Hz)')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Segment index $k$')
            plt.title(rf"Spectral Energy Map — Iteration {label}")
            plt.tight_layout()
            plt.savefig(os.path.join(spectral_dir, f"spectral_energy_map_{label}.pdf"))
            plt.close()

            if disorder_scale == 0:
                ####### Homogeneous TL ###########################################################
                t_ds_us = t_ds * 1e6  # convert once
                t_ds_trimmed = t_ds_us  # if not trimming
                v_map_ds = v_map_accum[::step, :]
                extent = [0, N * position_unit, t_ds_us[0], t_ds_us[len(v_map_ds) - 1]]
                log_map = np.log10(v_map_ds + log_eps)
                ####### Homogeneous TL ###########################################################

            # Heat map of ensemble-averaged |v_k(t)|^2
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(log_map, extent=extent, aspect='auto',
                        cmap='hot', origin='lower', rasterized=True)
            ax.set_xlabel(r"$k$, Position along TL (arbitrary units)")
            ax.set_ylabel(r"Time ($\mu$s)")
            if disorder_scale == 0:
                ax.set_ylim(0, ToF * 1.1 * 1e6) # Homogeneous TL
            ax.set_title(rf"Ensemble-Averaged $\log_{{10}}(\langle |v_k(t)|^2 - |v_k(0)|^2 \rangle)$ — Iteration {label}")
            fig.colorbar(im, ax=ax, label=r"$\log_{10}(\langle |v_k(t)|^2 \rangle)$")
            plt.tight_layout()

            # Save current heatmap
            plt.savefig(os.path.join(temporal_dir, f"temporal_energy_map_{label}.pdf"))
            plt.close(fig)

    # After the ensemble loop
    print("Simulation complete. Generating frames...")
    v_map_accum_ds = v_map_accum[::step, :]
    generate_energy_frames(v_map_accum_ds, t_ds, V_k_accum, f, f0, sigma_f, f0_factor, mfp, ell,
                       disorder_scale, frame_dir, time_range=(0, ToF * 1.1), n_frames=501)
    print("All frames saved to 'frames/' directory.")