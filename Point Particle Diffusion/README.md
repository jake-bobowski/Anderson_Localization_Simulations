# Point-Particle Clustering Simulation

This repository contains Python code to simulate the clustering of non-interacting point particles moving in a two-dimensional domain under the influence of a spatially disordered drift field.  

The model provides an accessible analogy to Anderson localization: as disorder in the drift field is increased, particle motion transitions from free diffusion to confinement and clustering.

---

## Features

- Simulates **N** particles on a square domain with periodic boundary conditions.
- Incorporates both **drift bias** (controlled by `mu_std`) and **diffusion** (set by `total_std`).
- Tracks quantitative measures over time:
  - ⟨r²⟩ (mean squared displacement)
  - Mean nearest-neighbor distance
  - Correlation between divergence of drift field and excess density
- Produces publication-quality figures (`.png`, 600 dpi by default).
- Saves all raw data (`.txt`, `.csv`) for reproducibility.
- Optional support for saving vector graphics (`.pdf`) of subplots.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/point_particle_clustering.git
cd point_particle_clustering
pip install -r requirements.txt
```

## Usage

Run the simulation from the command line:

```bash
python simulate_clustering.py --N 10000 --num_steps 1000 --mu_std 0.01 --total_std 0.02 --M 50
```


## Citation

If you use this code in academic work, please cite:

Jake S. Bobowski, "Point-particle clustering simulation code," GitHub repository,
https://github.com/yourusername/point_particle_clustering, 2025.
