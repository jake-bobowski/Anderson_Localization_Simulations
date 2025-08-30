# Particle Clustering Simulation with Spatially Disordered Drift Fields

This Python simulation models particle motion in a 2D periodic domain with spatially varying drift (bias) and diffusion. The drift field is disordered with a tunable standard deviation `mu_std`, allowing users to explore clustering phenomena and analogs to Anderson localization.

The simulation tracks metrics over time:
- Mean squared displacement ⟨r²⟩
- Mean nearest neighbor distance
- Correlation between divergence of the drift field and excess density

## Requirements

This code requires the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `tqdm`

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation using:

```bash
python RandomWalk_AL_GitHub.py --mu_std 0.01 --num_steps 1000
```

This will simulate 10,000 particles for 1000 time steps with a drift field disorder strength of 0.01.

### Optional Command-Line Arguments

| Argument         | Default   | Description                                |
|------------------|-----------|--------------------------------------------|
| `--N`            | 10000     | Number of particles                        |
| `--num_steps`    | 1000      | Number of simulation steps                 |
| `--mu_std`       | 0.01      | Std. dev. of drift field (disorder strength) |
| `--total_std`    | 0.02      | Total std. dev. from drift and diffusion   |
| `--M`            | 50        | Number of grid cells per side (M × M total) |

## Output

Simulation results are saved in a directory named `results_paper/`:

- `*_main.png`: Main 2D figure showing drift, divergence field, and particles
- `*_r2.png`: ⟨r²⟩ vs. step
- `*_nn.png`: Mean nearest neighbor distance vs. step
- `*_corr.png`: Divergence × excess density correlation vs. step
- `*_summary.csv`: Final average values of metrics
- `*_r2.txt`, `*_nn.txt`, `*_corr.txt`: Full time series for each metric
- `*_parameters.txt`: Record of simulation parameters used

## Citation

If you use this simulation or results derived from it, please cite:

```
@misc{clustering2025,
  author = {Jake S. Bobowski},
  title = {Particle clustering with spatially disordered drift fields},
  year = {2025},
  howpublished = {\url{https://github.com/jake-bobowski/Anderson_Localization_Simulations}},
}
```

---

© 2025 Jake Bobowski. Released under the MIT License.
