import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

# Derivatives
def derivative(u, v, dx, dy):
    du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)
    dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dx)
    dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dy)
    d2u_dx2 = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / (dx**2)
    d2u_dy2 = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / (dy**2)
    return du_dx, du_dy, dv_dx, dv_dy, d2u_dx2, d2u_dy2

# Turbulence scales
def turbulence_scales(u, v, nu, lengthscale):
    u_rms = np.sqrt(np.sum(u**2 + v**2) / (L_x * L_y))
    epsilon = u_rms**3 / lengthscale
    Re = u_rms * lengthscale / nu
    eta = (nu**3 / epsilon)**(1/4)
    tau_eta = (nu / epsilon)**(1/2)
    u_eta = (nu * epsilon)**(1/4)
    return Re, eta, tau_eta, u_eta

# Normalized Kinetic energy
def kinetic_energy(u, v):
    k_mean = np.sum((u**2 + v**2) / 2) / (L_x * L_y)
    k_normal = ((u**2 + v**2) / 2) / k_mean
    return k_normal
    
# Vorticity and Normalized Vorticity
def vorticity(u, v, dx, dy):
    _, du_dy, dv_dx, _, _, _ = derivative(u, v, dx, dy)
    omega_z = dv_dx - du_dy
    # Normalization logic
    omega_mean = np.sqrt(np.mean(omega_z**2))
    omega_normal = omega_z / (omega_mean + 1e-12) # Added safeguard
    return omega_normal, omega_z

# PDF, CDF, Skewness and Kurtosis    
def function(x):
    mean = np.mean(x)
    std = np.std(x)
    normalized = (x - mean) / std
    flat = normalized.ravel()
    hist, bins = np.histogram(flat, bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    pdf = hist / (np.sum(hist) * bin_width)
    cdf = np.cumsum(pdf) * bin_width
    skewness = ((np.sum((x - mean)**3) / (L_x * L_y)) / (std**3))
    kurtosis =((np.sum((x - mean)**4) / (L_x * L_y)) / (std**4))
    return bin_centers, pdf, cdf, skewness, kurtosis

# Contour plots
def plot_contour(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 6))
    if data is local_Re:
        plt.imshow(data[:: skip, :: skip], vmax=1e2)
    elif data is normalized_omega_z:
        plt.imshow(data[:: skip, :: skip],vmin=-5, vmax=5)
    elif data is normalized_kinetic_energy:
        plt.imshow(data[:: skip, :: skip], vmax=5)
    else:
        plt.imshow(data[:: skip, :: skip])
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)

# PDF and CDF plots
def plot_pdf_cdf(bins, pdf, cdf, title_pdf, title_cdf, filename):
    # Calculate normal distribution for comparison
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std)**2)
    normal_cdf = np.cumsum(normal_pdf) * (x_values[1] - x_values[0])
    normal_cdf /= normal_cdf[-1]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(bins, pdf, label=title_pdf)
    plt.plot(x_values, normal_pdf, label='Normal Distribution', linestyle='--')
    plt.title(title_pdf)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.yscale('log')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if cdf is not None:
        plt.plot(bins, cdf, label=title_cdf)
        plt.plot(x_values, normal_cdf, label='Normal Distribution', linestyle='--')
    else:
        plt.text(0.5, 0.5, 'No CDF Data Provided', ha='center')
    plt.title(title_cdf)
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.yscale('log')
    plt.legend()
    
    plt.savefig(filename)

# Load data
data = np.load("isotropic1024_slice.npz" )
u = data["u"]
v = data["v"]
w = data["w"]

# Data Definition
L = 2 * np.pi
L_x, L_y = 1024, 1024
dx = dy = L / L_x
nu, lengthscale = 0.000185, 1.364
x_values = np.linspace(-4, 4, 101)
skip = 5
k_normal = np.zeros((L_x, L_y))
omega_z = np.zeros((L_x + 2, L_y + 2))
local_Re = np.zeros((L_x, L_y))
mean, std = 0, 1

u_r1 = u[0:128,:]
u_r2 = u[129:256,:]
u_r3 = u[257:512,:]

Re, eta, tau_eta, u_eta = turbulence_scales(u, v, nu, lengthscale)  # Calculate turbulence scales
du_dx, du_dy, dv_dx, dv_dy, d2u_dx2, d2u_dy2 = derivative(u, v, dx, dy)

# Calculate non-linear term, viscous term, and local Reynolds number
N_lin = u*du_dx + v*du_dy
Visc = nu*(d2u_dx2 + d2u_dy2)
local_Re = np.where(np.abs(Visc)> 0, np.abs(N_lin / Visc), 0)

normalized_kinetic_energy = kinetic_energy(u, v)    # Calculate normalized kinetic energy
normalized_omega_z, omega_z = vorticity(u, v, dx, dy)   # Calculate vorticity and normalized vorticity
enstrophy = omega_z**2  # Calculate enstrophy

# Statistical analysis
bins_u, pdf_u, cdf_u, skewness_u, kurtosis_u = function(u)
bins_v, pdf_v, cdf_v, skewness_v, kurtosis_v = function(v)
bins_w, pdf_w, cdf_w, skewness_w, kurtosis_w = function(w)
bins_omega_z, pdf_omega_z, cdf_omega_z, skewness_omega_z, kurtosis_omega_z = function(omega_z)
bins_du_dx, pdf_du_dx, cdf_du_dx, skewness_du_dx, kurtosis_du_dx = function(du_dx)
bins_dv_dy, pdf_dv_dy, cdf_dv_dy, skewness_dv_dy, kurtosis_dv_dy = function(dv_dy)
bins_enstrophy, pdf_enstrophy, cdf_enstrophy, skewness_enstrophy, kurtosis_enstrophy = function(enstrophy)
bin_u_r1, pdf_u_r1,_,_,_ = function(u_r1)
bin_u_r2, pdf_u_r2,_,_,_ = function(u_r2)
bin_u_r3, pdf_u_r3,_,_,_ = function(u_r3)

# Turbulence scales and statistics
print(f"Reynolds Number (Re): {Re}")
print(f"Kolmogorov Length Scale (eta): {eta}")
print(f"Kolmogorov Time Scale (tau_eta): {tau_eta}")
print(f"Kolmogorov Velocity Scale (u_eta): {u_eta}")
print(f"Grid Cell Size (dx): {dx}")
print(f"Grid Cell Size / Kolmogorov Length (dx / eta): {dx / eta}")

# Skewness and kurtosis
print("Skewness and kurtosis\nfor u-component:", skewness_u, kurtosis_u,
      "\nfor v-component:", skewness_v, kurtosis_v,
      "\nfor w-component:", skewness_w, kurtosis_w,
      "\nfor vorticity:", skewness_omega_z, kurtosis_omega_z, 
      "\nfor velocity gradient:\ndu_dx:", skewness_du_dx, kurtosis_du_dx,
      "\ndv_dy:", skewness_dv_dy, kurtosis_dv_dy, 
      "\nfor enstrophy:", skewness_enstrophy, kurtosis_enstrophy)

# Contour plots
plot_contour(normalized_kinetic_energy, 'Normalized Turbulent Kinetic Energy Field', 'X', 'Y', 'Normalized_Turbulent_kinetic_energy_contour.png')
plot_contour(normalized_omega_z, 'Normalized Vorticity', 'X', 'Y', 'Normalized_Vorticity.png')
plot_contour(local_Re, 'Local Reynolds number', 'X', 'Y', 'Local_Reynolds_number.png')

# Plot of PDF and CDF
plot_pdf_cdf(bins_u, pdf_u, cdf_u, 'PDF of u-component', 'CDF of u-component', 'PDF_CDF_u_component.png')
plot_pdf_cdf(bins_v, pdf_v, cdf_v, 'PDF of v-component', 'CDF of v-component', 'PDF_CDF_v_component.png')
plot_pdf_cdf(bins_w, pdf_w, cdf_w, 'PDF of w-component', 'CDF of w-component', 'PDF_CDF_w_component.png')
plot_pdf_cdf(bins_omega_z, pdf_omega_z, cdf_omega_z, 'PDF of vorticity', 'CDF of vorticity', 'PDF_CDF_vorticity.png')
plot_pdf_cdf(bins_du_dx, pdf_du_dx, cdf_du_dx, 'PDF of du/dx', 'CDF of du/dx', 'PDF_CDF_du_dx.png')
plot_pdf_cdf(bins_dv_dy, pdf_dv_dy, cdf_dv_dy, 'PDF of dv/dy', 'CDF of dv/dy', 'PDF_CDF_dv_dy.png')
plot_pdf_cdf(bins_enstrophy, pdf_enstrophy, cdf_enstrophy, 'PDF of enstrophy', 'CDF of enstrophy', 'PDF_CDF_enstrophy.png')
plot_pdf_cdf(bin_u_r1, pdf_u_r1, None, 'PDF of u velocity (Range 0-128)', '', 'PDF_u_velocity_range_0_128.png')
plot_pdf_cdf(bin_u_r2, pdf_u_r2, None, 'PDF of u velocity (Range 129-256)', '', 'PDF_u_velocity_range_129_256.png')
plot_pdf_cdf(bin_u_r3, pdf_u_r3, None, 'PDF of u velocity (Range 257-512)', '', 'PDF_u_velocity_range_257_512.png')
