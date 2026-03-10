# Turbulence & PIV Data Analysis

## Overview
This repository contains comprehensive analysis of turbulence and Particle Image Velocimetry (PIV) datasets, focusing on advanced fluid dynamics characterization. The project demonstrates proficiency in scientific computing, data analysis, and computational fluid dynamics (CFD).

## Project Objectives
- Analyze isotropic turbulent flow using high-resolution velocity field data
- Compute critical turbulence metrics and Kolmogorov scales
- Perform statistical analysis of velocity fluctuations and vorticity
- Derive insights from grid-level resolution and flow dynamics

## Key Turbulence Analysis Outputs

### Flow Characteristics
| Metric | Value |
|--------|-------|
| **Reynolds Number (Re)** | 7376.65 |
| **Kolmogorov Length Scale (η)** | 0.00171 |
| **Kolmogorov Time Scale (τ_η)** | 0.01587 s |
| **Kolmogorov Velocity Scale (u_η)** | 0.1080 |
| **Grid Cell Size (dx)** | 0.00614 |
| **Grid Resolution (dx/η)** | 3.58 |

### Statistical Moments (Skewness, Kurtosis)

**Velocity Components:**
- **u-component**: Skewness = 0.172, Kurtosis = 3.045
- **v-component**: Skewness = 0.091, Kurtosis = 2.591
- **w-component**: Skewness = -0.220, Kurtosis = 2.583

**Turbulence Properties:**
- **Vorticity**: Skewness = 0.315, Kurtosis = 12.617
- **Velocity Gradient (du/dx)**: Skewness = -0.599, Kurtosis = 7.597
- **Velocity Gradient (dv/dy)**: Skewness = -0.477, Kurtosis = 7.152
- **Enstrophy**: Skewness = 33.134, Kurtosis = 4385.617

## Repository Structure
```
Data-Analysis-/
├── PIV_data_anlaysis.py           # PIV data processing and analysis
├── Turbulence_data_analysis_1.py  # Initial turbulence metrics computation
├── Turbulence_data_analysis_2.py  # Advanced statistical analysis
├── Turbulence_data_analysis_3.py  # Comprehensive turbulence characterization
├── isotropic1024_slice.npz        # High-resolution velocity field dataset
├── PIV_results/                   # PIV analysis outputs
├── Turbulence_results/            # Turbulence analysis results
└── DATA/                          # Additional data directory
```

## Technologies & Skills Demonstrated
- **Languages**: Python
- **Scientific Computing**: NumPy, SciPy
- **Data Analysis**: Statistical analysis, signal processing
- **Domain Expertise:** 
  - Computational Fluid Dynamics (CFD)
  - Turbulence modeling and analysis
  - Flow visualization and characterization
  - Kolmogorov cascade theory

## Key Analyses Performed
1. **Kolmogorov Scale Computation** - Derived microscale properties from high-Reynolds number flow
2. **Statistical Moments Analysis** - Calculated skewness and kurtosis for velocity and vorticity fields
3. **Grid Resolution Assessment** - Evaluated DNS/LES grid adequacy relative to Kolmogorov scales
4. **Enstrophy Characterization** - Analyzed vorticity distribution and higher-order statistics

## Installation
```bash
git clone https://github.com/ak587/Data-Analysis-.git
cd Data-Analysis-
pip install numpy scipy matplotlib seaborn pandas
```

## Usage
Run individual analysis scripts to reproduce results:
```bash
python Turbulence_data_analysis_1.py
python Turbulence_data_analysis_2.py
python Turbulence_data_analysis_3.py
python PIV_data_anlaysis.py
```

Results are saved to `Turbulence_results/` and `PIV_results/` directories.

## What This Demonstrates
✓ **Advanced Technical Skills**: Proficiency in scientific computing and fluid dynamics  
✓ **Data Analysis Expertise**: Statistical characterization of complex datasets  
✓ **Scientific Rigor**: Proper computation of turbulence metrics following established theory  
✓ **Problem-Solving**: Multi-stage analysis pipeline for comprehensive flow characterization  
✓ **Python Mastery**: Well-structured code for scientific applications  

---

**Contact**: [Your Name] | [Your Email] | [LinkedIn Profile]