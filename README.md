# Anderson Localization Simulations

This repository contains two complementary simulations that explore Anderson localization — the suppression of transport in disordered systems due to interference and spatial disorder.

The goal is to provide visual and quantitative tools for understanding localization in both classical and wave-based transport. These simulations are suitable for research, teaching, and generating publication-quality figures and animations.

Together, the simulations illustrate localization from complementary perspectives: particle diffusion highlights suppressed transport statistically, while the transmission-line model demonstrates wave interference and energy confinement.

## Simulations

### 🔹 [`Point_Particle_Diffusion/`](Point_Particle_Diffusion)

A classical particle diffusion simulation with spatially disordered drift and diffusion fields. The model exhibits clustering and suppressed transport analogous to Anderson localization.

Includes:
- Particle motion visualizations
- ⟨r²⟩ vs time
- Anderson localization metrics

➡️ See [`Point_Particle_Diffusion/README.md`](Point_Particle_Diffusion/README.md) for usage and outputs.

---

### 🔹 [`Disordered_Transmission_Line/`](Disordered_Transmission_Line)

A wave-based model simulating electromagnetic pulse propagation through a 1D disordered transmission line. Using transfer matrices and ensemble averaging, the simulation reveals Anderson localization in both time and frequency domains and connects directly to experimental transmission-line systems.

Includes:
- Temporal energy maps ⟨|vₖ(t)|²⟩
- Spectral energy maps |Vₖ(f)|²
- Frame-by-frame animations

➡️ See [`Disordered_Transmission_Line/README.md`](Disordered_Transmission_Line/README.md) for usage and outputs.

---

## License

All simulation code and outputs are released under the [MIT License](LICENSE).

## Citation

If you use this repository or the results it produces, please cite:

```
@misc{AndersonSims2025,
  author = {Jake S. Bobowski},
  title = {Anderson Localization Simulations},
  year = {2026},
  howpublished = {\url{https://github.com/jake-bobowski/Anderson_Localization_Simulations}},
}
```

---

© 2026 Jake Bobowski — University of British Columbia
