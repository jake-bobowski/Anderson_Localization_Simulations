# 1D Transmission Line Simulation of Anderson Localization

This Python simulation models the propagation of an electromagnetic pulse through a one-dimensional transmission line (TL) with spatially disordered impedance and propagation speed. By varying the disorder strength, the simulation explores how multiple scattering and interference lead to Anderson localization.

The simulation computes both time-domain and frequency-domain metrics using ensemble averaging over thousands of disordered realizations. Outputs include:

- Spatially resolved energy density ⟨|v_k(t)|²⟩ over time
- Spectral energy maps |V_k(f)|² vs frequency
- Heatmaps and frame-by-frame evolution of energy transport

## Requirements

This code requires the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation using:

```bash
python TL_AL_GitHub.py --N_segments 500 --N_iter 5000 --disorder_scale 0.5
```

This will simulate 500-segment disordered transmission lines with disorder strength 0.5, averaged over 5000 realizations.

### Optional Command-Line Arguments

| Argument           | Default  | Description                                                   |
|--------------------|----------|---------------------------------------------------------------|
| `--N_segments`     | 500      | Number of TL segments                                         |
| `--n_freq`         | 1048576  | Number of frequency points (must be a power of 2)             |
| `--N_iter`         | 5000     | Number of ensemble iterations                                 |
| `--mfp`            | 0.15     | Mean free path (m), sets disorder correlation length          |
| `--disorder_scale` | 0.5      | Standard deviation of C, L disorder (relative to mean)        |
| `--disorder_onset` | 0.2      | Fraction of TL length over which disorder gradually turns on  |
| `--f0_factor`      | 2.0      | f₀ = f0_factor × v₀ / mfp; center freq. of input Gaussian pulse |
| `--sigma_divisor`  | 20.0     | Standard deviation of input pulse = f₀ / sigma_divisor        |
| `--output_dir`     | heatmaps_avg | Folder for all output files                                |
| `--save_interval`  | 100      | Save plots every N iterations during simulation               |

## Output

Simulation results are saved in the directory specified by `--output_dir` (default: `heatmaps_avg/`). Key outputs include:

- `parameters.txt`: Full record of all simulation parameters
- `V_k_f0_energy.txt`: Final ensemble-averaged energy per segment at f₀
- `E_total_vs_time.txt`: Normalized total energy vs time

**Subdirectories:**

- `temporal/`: 
  - `temporal_energy_map_final.pdf`: Final time-domain energy heatmap
  - `v_k_time_window_###.txt`: Time-binned spatial energy snapshots

- `spectral/`: 
  - `spectral_energy_map_final.pdf`: Final spectral energy map
  - Shows |V_k(f)|² vs frequency for each segment

- `frames/`: 
  - `frame_###_tXXXns.png`: Animation frames showing ⟨|v_k(t)|²⟩ and |V_k(f)|²
  - `*_time.txt` and `*_freq.txt`: Raw data for each frame

## Citation

If you use this simulation or results derived from it, please cite:

```
@misc{TL_AL_2025,
  author = {Jake S. Bobowski},
  title = {1D transmission line simulation of Anderson localization},
  year = {2025},
  howpublished = {\url{https://github.com/jake-bobowski/Anderson_Localization_Simulations}},
}
```

---

© 2025 Jake Bobowski. Released under the MIT License.
