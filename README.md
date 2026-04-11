# High-Resolution Turbulence Data Engine & Lagrangian Tracking

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Matrix%20Computing-NumPy-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/Interpolation-SciPy-lightblue.svg)](https://scipy.org/)
[![FFT](https://img.shields.io/badge/Math-Fast%20Fourier%20Transforms-yellow.svg)]()

## The Computational Challenge
Analyzing Direct Numerical Simulation (DNS) data of Homogeneous Isotropic Turbulence (HIT) presents a massive "Big Data" challenge in mechanical engineering. 

Processing a **$1024x1024x3$ grid (over 3 Million volumetric cells)** using standard nested `for` loops in Python will take hours to execute. This repository houses a **highly-optimized, fully vectorized Python analytics engine** designed to ingest large DNS arrays, compute complex tensor mathematics, and execute Lagrangian particle tracking in minutes.

---

## Core Analytics Modules

The engine is modularized into three distinct physics-data pipelines:

### 1. Eulerian Field Processing & Tensor Calculus (`Turbulence_data_analysis-1.py`)
Extracts fundamental flow topology from raw velocity fields without relying on commercial software.
* **Velocity Gradient Tensor:** Computes spatial derivatives ($\partial u_i / \partial x_j$) across the 1024x1024x3 domain using vectorized finite differencing.
* **Flow Topology:** Calculates the invariant $P, Q, R$ matrices to classify flow regions (e.g., strain-dominated vs. rotation-dominated) via **Q-R scatter diagrams**.
* **Statistical Intermittency:** Evaluates non-Gaussian behavior in turbulence via high-order moments (Skewness/Kurtosis) of enstrophy and velocity gradients.

### 2. Spectral Dynamics & Energy Cascades (`Turbulence_data_analysis-2.py`)
Validates the energy transfer mechanisms across multiple scales using Frequency Domain analysis.
* **Fast Fourier Transforms (FFT):** Converts spatial velocity fields into 1D and 2D energy spectra ($E(k)$).
* **Kolmogorov Validation:** Algorithmically fits theoretical $-5/3$ power laws to the inertial sub-range to verify energy cascade physics.
* **Anomalous Scaling:** Computes structure functions up to the 7th order to quantify internal intermittency and deviations from classical K41 theory.

### 3. Vectorized Lagrangian Particle Tracking (`Turbulence_data_analysis-3.py`)
A custom-built, headless particle tracking algorithm to study chaotic dispersion.
* **Massive Scale Tracking:** Simulates the trajectory of **$10^4$ individual fluid particles** simultaneously.
* **Sub-Grid Interpolation:** Utilizes SciPy's `RectBivariateSpline` for continuous velocity field interpolation, allowing particles to move seamlessly off-grid.
* **Lyapunov Exponents & Richardson's Law:** Measures the exponential separation of particle pairs to quantify the chaotic nature (predictability horizon) of the flow.

---

## Extracted Physics & Visualizations

> [Results](Results)

### 1. Flow Topology (Q-R Discriminant Plane)
Visualizes the invariant map of the velocity gradient tensor. The "teardrop" shape strictly aligns with theoretical turbulence topology, separating focal regions from saddle/strain zones.
> ![Q_R_scatter.png](Results/Turbulence_data_analysis-3/Q_R_scatter.png)

### 2. The Energy Cascade (Spectral Density)
Automated FFT mapping of the kinetic energy spectrum, clearly identifying the integral scale, the $-5/3$ inertial sub-range, and the dissipation scales.
> ![1D_Energy_Spectrum](Results/Turbulence_data_analysis-2/1D_Energy_Spectrum.png)

### 3. Lagrangian Particle Tracking
Tracks 10,000+ fluid particles simultaneously to characterize turbulent flow structures.
> ![Lagrangian Particle Tracking](Results/Turbulence_data_analysis-3/particle_trajectories_1000.png)

---

## ⚙️ Computational Performance & Tech Stack

* **Memory Management:** Utilized NumPy broadcasting and array slicing to eliminate `for` loops during multi-million point calculations.
* **Mathematical Operations:** `np.fft.fftn` for spectral transformations, `np.roots` for localized eigenvalue extraction of the gradient tensor.
