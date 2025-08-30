# Anderson Localization Simulations

This repository contains two complementary simulations that explore Anderson localization — the suppression of wave diffusion in disordered systems due to interference.

The goal is to provide visual and quantitative tools for understanding localization in both classical and wave-based transport systems. These simulations are suitable for research, teaching, or generating high-quality figures and animations for publication.

## Simulations

### 🔹 [`Point_Particle_Random_Walk/`](Point_Particle_Random_Walk)

A classical particle diffusion simulation with spatially disordered drift and diffusion fields. Tracks clustering, nearest neighbor statistics, and divergence-density correlations.

Includes:
- Particle motion visualizations
- ⟨r²⟩ vs time
- Anderson localization metrics

➡️ See [`Point_Particle_Random_Walk/README.md`](Point_Particle_Random_Walk/README.md) for usage and outputs.

---

### 🔹 [`Transmission_Line_Simulation/`](Transmission_Line_Simulation)

A wave-based model simulating electromagnetic pulse propagation through a 1D disordered transmission line. Uses transfer matrices and ensemble averaging to reveal localization in both time and frequency domains.

Includes:
- Temporal energy maps ⟨|vₖ(t)|²⟩
- Spectral energy maps |Vₖ(f)|²
- Frame-by-frame animations

➡️ See [`Transmission_Line_Simulation/README.md`](Transmission_Line_Simulation/README.md) for usage and outputs.

---

## License

All simulation code and outputs are released under the [MIT License](LICENSE).

## Citation

If you use this repository or the results it produces, please cite:

```
@misc{AndersonSims2025,
  author = {Jake S. Bobowski},
  title = {Anderson Localization Simulations},
  year = {2025},
  howpublished = {\url{https://github.com/jake-bobowski/Anderson_Localization_Simulations}},
}
```

---

© 2025 Jake Bobowski — University of British Columbia
