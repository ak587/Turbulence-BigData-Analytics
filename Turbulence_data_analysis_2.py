#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = np.load("isotropic1024_slice.npz" )
u = data["u"]
v = data["v"]
w = data["w"]

# Data
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

r_values = []
r_length = []
r_val = []
for i in range(50):
    r_value = int(0 + (512 - 0) * (math.exp(i / 49) - 1) / (math.exp(1) - 1))
    r_values.append(r_value)
    r_value_with_pi = r_value * math.pi / 512
    r_length.append(r_value_with_pi)
    r_val_temp = int(i * (1024 / 49))
    r_val.append(r_val_temp)


# In[3]:


# Verify Parseval's Theorem
u0 = u[:,0]
u0_fft = np.fft.fft(u0)
total_energy_spatial_y0 = np.sum(np.abs(u0)**2)
total_energy_spectral_y0 = np.sum(np.abs(u0_fft)**2) / Nx
print(f"Total energy in spatial domain: {total_energy_spatial_y0}\nTotal energy in spectral domain: {total_energy_spectral_y0}")


# In[5]:


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


# In[7]:


# Calculation of 1D Energy spectrum
for y in range(Ny):
    u_fft_y = np.fft.fft(u[:,y])
    v_fft_y = np.fft.fft(v[:,y])
    E_k_y[:,y] = 0.5 * (np.abs(u_fft_y)**2 + np.abs(v_fft_y)**2)

E_k = np.mean(E_k_y, axis=1)
E_k = np.fft.fftshift(E_k)
E_k = E_k[Nx//2:]

E_k_fit, E_k_ref, alpha, inertial_range = fit(k, E_k)


# In[65]:


# plot for 1D Energy Spectrum
plt.loglog(k, E_k)
plt.loglog(k[inertial_range], E_k_fit[inertial_range], 'r--', label=f"Fit: $k^{alpha}$")
plt.loglog(k[inertial_range], E_k_ref[inertial_range], 'k-.', label=r"$k^{-5/3}$ reference")
plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"Energy Spectrum $E(k)$")
plt.legend()
plt.grid()
plt.title("1D Energy Spectrum")
plt.show()


# In[11]:


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


# In[69]:


# Plot 2D Energy Spectrum
plt.loglog(k_bin, E_k_shell)
plt.loglog(k_bin[inertial_range_shell], E_k_fit_shell[inertial_range_shell], 'r--', label=f"Fit: $k^{alpha_shell}$")
plt.loglog(k_bin[inertial_range_shell], E_k_ref_shell[inertial_range_shell], 'k-.', label=r"$k^{-5/3}$ reference")
plt.xlabel(r"Wavenumber $k$")
plt.ylabel(r"Energy Spectrum $E(k)$")
plt.legend()
plt.grid()
plt.title("2D Energy Spectrum")
plt.show()


# In[19]:


def correlation(u, r_val):
    term = 0
    R = np.zeros(len(r_val))
    for r in r_val:
        count = 0
        for x in range(Nx):
            for y in range(Ny):
                # Apply periodic boundary condition for x
                x_periodic = (x + r) % Nx
                R[term] += u[x, y] * u[x_periodic, y]
                count += 1
        print(r, term, R[term])
        R[term] = R[term] / count if count > 0 else 0
        term += 1   
    return R / R[0]
R_l = correlation(u, r_val)
R_t = correlation(v, r_val)


# In[87]:


# Velocity correlation function
plt.plot(r_val, R_l, label="Longitudinal correlation")
plt.plot(r_val, R_t, label="Transverse correlation")
plt.xlabel("r(in terms of grids)")
plt.ylabel("Correlation coefficient")
plt.legend()
plt.grid()
plt.title("Velocity correlation function")
plt.show()


# In[24]:


# Sturcture functions
def structure(u, r_values, p):
    S_p = np.zeros(len(r_values))
    for r in range(len(r_values)):
        count = 0
        for x in range(0, Nx):
            for y in range(0, Ny):
                x_periodic = (x + r) % Nx
                S_p[r] += np.abs(u[x_periodic, y] - u[x, y])**p
                count += 1
        S_p[r] = S_p[r] / count if count > 0 else 0
        print(r)
    print("----------------Done----------------")
    return S_p
S_1 = structure(u, r_values, 1)
S_2 = structure(u, r_values, 2)
S_3 = structure(u, r_values, 3)
S_4 = structure(u, r_values, 4)
S_5 = structure(u, r_values, 5)
S_6 = structure(u, r_values, 6)
S_7 = structure(u, r_values, 7)


# In[89]:


# Plot Structure function S_p vs r
plt.loglog(r_values, S_1, label="S_1")
plt.loglog(r_values, S_2, label="S_1")
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
plt.show()


# In[29]:


# Kolmogorove's 4/5th law
S_3_kolmogorov = np.zeros(len(r_length))
term = 0
for r in r_length:
    S_3_kolmogorov[term] = (4 * epsilon * r) / 5
    term += 1


# In[101]:


# Plot Kolmogorove's 4/5th law
plt.loglog(r_length, S_3, label="S_3")
plt.loglog(r_length, S_3_kolmogorov, label="S_3_kolmogorov")
plt.xlabel("r(in terms of grid)")
plt.ylabel("S_3")           
plt.legend()
plt.title("Kolmogorove's 4/5th law")
plt.grid()
plt.show()


# In[103]:


# Plot Extended Self-Similarity (ESS)
plt.loglog(S_3, S_1, label="S_1")
plt.loglog(S_3, S_2, label="S_1")
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
plt.show()


# In[130]:


p_values = np.arange(1, 8)
tau = []
tau_errors = []

def taufit(r_values, S_p, order=None):
    mask = (S_p > 0) & (np.array(r_values) > 0)
    valid_points = np.sum(mask)
    log_S_p = np.log(S_p[mask])
    log_r_values = np.log(np.array(r_values)[mask])
    slope, intercept, r_value, p_value, slope_err = stats.linregress(log_r_values, log_S_p)
    return slope, slope_err

structure_functions = [S_1, S_2, S_3, S_4, S_5, S_6, S_7]
for i, S_p in enumerate(structure_functions):
    p = i + 1  # p value is index + 1
    tau_p, tau_p_err = taufit(r_values, S_p, order=p)
    tau.append(tau_p)
    tau_errors.append(tau_p_err)

tau_p_theoretical = [p/3 for p in p_values]


# In[132]:


# Structure Function Scaling Exponents
plt.errorbar(valid_p, valid_tau, yerr=valid_errors, fmt='o-', 
             capsize=5, elinewidth=1.5, capthick=1.5, 
             label='Measured tau_p', markersize=8)

plt.plot(p_values, tau_p_theoretical, 's--', label='Theoretical tau_p = p/3')
plt.xlabel('Order (p)')
plt.ylabel('Scaling Exponent (tau_p)')
plt.title('Structure Function Scaling Exponents')
plt.grid()
plt.legend()
plt.show()


# In[ ]:




