import numpy as np
import math
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats

# Function to calculate 1D Energy Spectrum
def fit(k, E_k):
    inertial_range = (k > (2 * np.pi * 6 / L)) & (k < 2 * np.pi / (60 * eta))
    log_E_k = np.log(E_k[inertial_range])
    log_k = np.log(k[inertial_range])
    alpha, log_A = np.polyfit(log_k, log_E_k, 1)
    A = np.exp(log_A)
    E_k_fit = A * k**(alpha)
    A_ref = E_k[inertial_range][0] / k[inertial_range][0]**(-5/3)
    E_k_ref = A_ref * k**(-5/3)
    return E_k_fit, E_k_ref, alpha, inertial_range

def correlation(u, r_val):
    R = np.array([np.mean(u*np.roll(u, -r, axis=0)) for r in r_val])
    return R / R[0]

# Sturcture functions
def structure(u, r_values, p):
    S_p = np.array([np.mean(np.abs((np.roll(u, -r, axis=0)-u))**p) for r in r_values])
    return S_p

def taufit(r_values, S_p):
    log_S_p = np.log(S_p)
    log_r_values = np.log(np.array(r_values))
    slope, intercept, r_value, p_value, slope_err = stats.linregress(log_r_values, log_S_p)
    return slope, slope_err

# Load data
data = np.load("isotropic1024_slice.npz" )
u = data["u"]
v = data["v"]
w = data["w"]

# Data Definition
L = 2 * np.pi
Nx, Ny = 1024, 1024
dx = L / Nx
dy = L / Ny
nu = 0.000185
E_k_y = np.zeros((Nx, Ny))
k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
k = np.fft.fftshift(k)
k = k[Nx//2:]
skip = 5

u_rms = np.sqrt(np.mean(u**2 + v**2))
epsilon = u_rms**3 / (1.364)
eta = (nu**3 / epsilon)**(1/4)

r_values = np.unique(np.logspace(0,np.log10(512),50).astype(int))
r_length = r_values * np.pi / 512
r_val = np.linspace(0, 1024, 50)

R_l = correlation(u, r_val)
R_t = correlation(v, r_val)

# Verify Parseval's Theorem
u0 = u[:,0]
u0_fft = np.fft.fft(u0)
total_energy_spatial_y0 = np.sum(np.abs(u0)**2)
total_energy_spectral_y0 = np.sum(np.abs(u0_fft)**2) / Nx
print(f"Total energy in spatial domain: {total_energy_spatial_y0}\nTotal energy in spectral domain: {total_energy_spectral_y0}")

# Calculation of 1D Energy spectrum
u_fft_all = np.fft.fft(u, axis=0)
v_fft_all = np.fft.fft(v, axis=0)
E_k_y = 0.5 * (np.abs(u_fft_all)**2 + np.abs(v_fft_all)**2)

E_k = np.mean(E_k_y, axis=1)
E_k = np.fft.fftshift(E_k)[Nx//2:]

E_k_fit, E_k_ref, alpha, inertial_range = fit(k, E_k)

# Calculate 2D Energy Spectrum
u_fft = np.fft.fft2(u)
v_fft = np.fft.fft2(v)

E_kxky = (np.abs(u_fft)**2 + np.abs(v_fft)**2) * 0.5
E_kxky = np.fft.fftshift(E_kxky)

k_x = np.fft.fftfreq(Nx, dx) * 2 * np.pi
k_x = np.fft.fftshift(k_x)
k_y = np.fft.fftfreq(Ny, dy) * 2 * np.pi
k_y = np.fft.fftshift(k_y)
k_x, k_y = np.meshgrid(k_x, k_y)
k_shell = np.sqrt(k_x**2 + k_y**2)

k_bin = np.arange(0.5, np.max(k_shell), 1)
E_k_shell = np.zeros((len(k_bin)))
N = np.zeros((len(k_bin)))

for i in range(len(k_x)):
    for j in range(len(k_y)):
        k_ij = k_shell[i,j]
        bin_ij = np.digitize(k_ij, k_bin) - 1
        if 0 <= bin_ij <= len(E_k):
            E_k_shell[bin_ij] += E_kxky[i,j]
            N[bin_ij] += 1

E_k_shell[N > 0] /= N[N > 0]

E_k_fit_shell, E_k_ref_shell, alpha_shell, inertial_range_shell = fit(k_bin, E_k_shell)

S_1 = structure(u, r_values, 1)
S_2 = structure(u, r_values, 2)
S_3 = structure(u, r_values, 3)
S_4 = structure(u, r_values, 4)
S_5 = structure(u, r_values, 5)
S_6 = structure(u, r_values, 6)
S_7 = structure(u, r_values, 7)

p_values = np.arange(1, 8)
tau = []
tau_errors = []

structure_functions = [S_1, S_2, S_3, S_4, S_5, S_6, S_7]
for i, S_p in enumerate(structure_functions):
    tau_p, tau_p_err = taufit(r_values, S_p)
    tau.append(tau_p)
    tau_errors.append(tau_p_err)

tau_p_theoretical = [p/3 for p in p_values]

# Structure Function Scaling Exponents
p_values = np.array(p_values)
tau = np.array(tau)
tau_errors = np.array(tau_errors)

# Kolmogorove's 4/5th law
S_3_kolmogorov = np.zeros(len(r_length))
for i, r in enumerate(r_length):
    S_3_kolmogorov[i] = (4 * epsilon * r) / 5

# plot for 1D Energy Spectrum
plt.figure(figsize=(8, 6))
plt.loglog(k, E_k)
plt.loglog(k[inertial_range], E_k_fit[inertial_range], 'r--', label=f"Fit: $k^{alpha}$")
plt.loglog(k[inertial_range], E_k_ref[inertial_range], 'k-.', label=r"$k^{-5/3}$ reference")
plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"Energy Spectrum $E(k)$")
plt.legend()
plt.grid()
plt.title("1D Energy Spectrum")
plt.savefig("1D_Energy_Spectrum.png", dpi=300)

# Plot 2D Energy Spectrum
plt.figure(figsize=(8, 6))
plt.loglog(k_bin, E_k_shell)
plt.loglog(k_bin[inertial_range_shell], E_k_fit_shell[inertial_range_shell], 'r--', label=f"Fit: $k^{alpha_shell}$")
plt.loglog(k_bin[inertial_range_shell], E_k_ref_shell[inertial_range_shell], 'k-.', label=r"$k^{-5/3}$ reference")
plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"Energy Spectrum $E(k)$")
plt.legend()
plt.grid()
plt.title("2D Energy Spectrum")
plt.savefig("2D_Energy_Spectrum.png", dpi=300)

# Velocity correlation function
plt.figure(figsize=(8, 6))
plt.plot(r_val, R_l, label="Longitudinal correlation")
plt.plot(r_val, R_t, label="Transverse correlation")
plt.xlabel("r(in terms of grids)")
plt.ylabel("Correlation coefficient")
plt.legend()
plt.grid()
plt.title("Velocity correlation function")
plt.savefig("Velocity_correlation_function.png", dpi=300)

# Plot Structure function S_p vs r
plt.figure(figsize=(8, 6))
plt.loglog(r_values, S_1, label="S_1")
plt.loglog(r_values, S_2, label="S_2")
plt.loglog(r_values, S_3, label="S_3")
plt.loglog(r_values, S_4, label="S_4")
plt.loglog(r_values, S_5, label="S_5")
plt.loglog(r_values, S_6, label="S_6")
plt.loglog(r_values, S_7, label="S_7")
plt.xlabel("r(in terms of grids)")
plt.ylabel("S_p")           
plt.legend()
plt.title("Structure function S_p vs r")
plt.grid()
plt.savefig("Structure_function_S_p_vs_r.png", dpi=300)

# Plot Kolmogorove's 4/5th law
plt.figure(figsize=(8, 6))
plt.loglog(r_length, S_3, label="S_3")
plt.loglog(r_length, S_3_kolmogorov, label="S_3_kolmogorov")
plt.xlabel("r(in terms of grid)")
plt.ylabel("S_3")           
plt.legend()
plt.title("Kolmogorove's 4/5th law")
plt.grid()
plt.savefig("Kolmogorov_4_5th_law.png", dpi=300)

# Plot Extended Self-Similarity (ESS)
plt.figure(figsize=(8, 6))
plt.loglog(S_3, S_1, label="S_1")
plt.loglog(S_3, S_2, label="S_2")
plt.loglog(S_3, S_3, label="S_3")
plt.loglog(S_3, S_4, label="S_4")
plt.loglog(S_3, S_5, label="S_5")
plt.loglog(S_3, S_6, label="S_6")
plt.loglog(S_3, S_7, label="S_7")
plt.xlabel("S_3")
plt.ylabel("S_p")           
plt.legend()
plt.title("Extended Self-Similarity (ESS)")
plt.grid()
plt.savefig("Extended_Self-Similarity.png", dpi=300)

plt.figure(figsize=(10, 6))
plt.errorbar(p_values, tau, yerr=tau_errors, fmt='o-', 
             capsize=5, elinewidth=1.5, capthick=1.5, 
             label='Measured tau_p', markersize=8)

plt.plot(p_values, tau_p_theoretical, 's--', label='Theoretical tau_p = p/3')
plt.xlabel('Order (p)')
plt.ylabel('Scaling Exponent (tau_p)')
plt.title('Structure Function Scaling Exponents')
plt.grid()
plt.legend()
plt.savefig(f"Structure Function Scaling Exponents.png")

# Rough work
# u0 = u[:, 0]
# u0_fft = np.fft.fft(u0)
# print(f"u0________________\n{u0}\nu0fft_____________\n{u0_fft}")
# plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.plot(np.linspace(0, 1024, 1024), u0)
# plt.title("u0")
# plt.subplot(1, 2, 2)
# plt.plot(np.linspace(0, 1, 1024), np.abs(u0_fft))
# plt.title("u0_fft")
# plt.show()