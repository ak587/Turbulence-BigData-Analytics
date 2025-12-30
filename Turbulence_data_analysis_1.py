#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
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
L_x, L_y = 1024, 1024
dx = L / L_x
nu = 0.000185
x_values = np.linspace(-4, 4, 101)
skip = 1
k_normal = np.zeros((L_x, L_y))
omega_z = np.zeros((L_x + 2, L_y + 2))
local_Re = np.zeros((L_x, L_y))
mean, std = 0, 1

# Normalized Kinetic energy
def kinetic_energy(u, v):
    k_mean = np.sum((u**2 + v**2) / 2) / (L_x * L_y)
    k_normal = ((u**2 + v**2) / 2) / k_mean
    return k_normal
    
# Vorticity and Normalized Vorticity
def vorticity(u, v, L_x, L_y, dx):
    padded_u = np.zeros((1026, 1026))
    padded_v = np.zeros((1026, 1026))
    
    padded_u[1:-1, 1:-1] = u
    padded_u[0, 1:-1] = u[-1, :]
    padded_u[-1, 1:-1] = u[0, :]
    padded_u[1:-1, 0] = u[:, -1]
    padded_u[1:-1, -1] = u[:, 0]
    
    padded_v[1:-1, 1:-1] = v
    padded_v[-1, 1:-1] = v[0, :]
    padded_v[1:-1, 0] = v[:, -1]
    padded_v[1:-1, -1] = v[:, 0]
    
    for i in range(0, L_x + 1):
        for j in range(0, L_y + 1):
            omega_z[i+1,j+1] = (((padded_v[i, j+1] - padded_v[i, j-1]) - (padded_u[i+1, j] - padded_u[i-1, j])) / (2 * dx))
    
    omega_mean = np.sqrt(np.sum(omega_z**2) / (L_x * L_y))
    omega_normal = omega_z / omega_mean
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
    pdf_interpolated = np.interp(x_values, bin_centers, pdf)
    cdf = np.cumsum(pdf) * bin_width
    skewness = ((np.sum((x - mean)**3) / (L_x * L_y)) / (std**3))
    kurtosis =((np.sum((x - mean)**4) / (L_x * L_y)) / (std**4))
    return bin_centers, pdf, cdf, skewness, kurtosis

def derivative(u, v, dx):
    u_plus = np.roll(u, -1, axis=1)
    u_minus = np.roll(u, 1, axis=1)
    
    v_plus = np.roll(v, -1, axis=0)
    v_minus = np.roll(v, 1, axis=0)
    du_dx = (u_plus - u_minus) / (2 * dx)
    dv_dy = (v_plus - v_minus) / (2 * dx)
    return du_dx, dv_dy

# Question 2.1
u_rms = np.sqrt(np.sum(u**2 + v**2) / (L_x * L_y))

epsilon = u_rms**3 / 1.364
Re = u_rms * 1.364 / nu
eta = (nu**3 / epsilon)**(1/4)
tau_eta = (nu / epsilon)**(1/2)
u_eta = (nu * epsilon)**(1/4)

print(f"Reynolds Number (Re): {Re}")
print(f"Kolmogorov Length Scale (eta): {eta}")
print(f"Kolmogorov Time Scale (tau_eta): {tau_eta}")
print(f"Kolmogorov Velocity Scale (u_eta): {u_eta}")
print(f"Grid Cell Size (dx): {dx}")
print(f"Grid Cell Size / Kolmogorov Length (dx / eta): {dx / eta}")

# Question 2.2
normalized_kinetic_energy = kinetic_energy(u, v)

# Question 2.3
normalized_omega_z, omega_z = vorticity(u, v, L_x, L_y, dx)
enstrophy = omega_z**2

# Question 2.4
du_dx, dv_dy = derivative(u, v, dx)
bins_u, pdf_u, cdf_u, skewness_u, kurtosis_u = function(u)
bins_v, pdf_v, cdf_v, skewness_v, kurtosis_v = function(v)
bins_w, pdf_w, cdf_w, skewness_w, kurtosis_w = function(w)
bins_omega_z, pdf_omega_z, cdf_omega_z, skewness_omega_z, kurtosis_omega_z = function(omega_z)
bins_du_dx, pdf_du_dx, cdf_du_dx, skewness_du_dx, kurtosis_du_dx = function(du_dx)
bins_dv_dy, pdf_dv_dy, cdf_dv_dy, skewness_dv_dy, kurtosis_dv_dy = function(dv_dy)
bins_enstrophy, pdf_enstrophy, cdf_enstrophy, skewness_enstrophy, kurtosis_enstrophy = function(enstrophy)

normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std)**2)
normal_cdf = np.cumsum(normal_pdf) * (x_values[1] - x_values[0])
normal_cdf /= normal_cdf[-1]

u_r1 = u[0:128,:]
u_r2 = u[129:256,:]
u_r3 = u[257:512,:]
bin_u_r1, pdf_u_r1,_,_,_ = function(u_r1)
bin_u_r2, pdf_u_r2,_,_,_ = function(u_r2)
bin_u_r3, pdf_u_r3,_,_,_ = function(u_r3)

print("Skewness and kurtosis\n for u-component\n", skewness_u, kurtosis_u,
      "\nfor v-component\n", skewness_v, kurtosis_v,
      "\nfor w-component\n", skewness_w, kurtosis_w,
      "\nfor vorticity:\n", skewness_omega_z, kurtosis_omega_z, 
      "\nfor velocity gradient:\ndu_dx\n", skewness_du_dx, kurtosis_du_dx,"\ndv_dy\n", skewness_dv_dy, kurtosis_dv_dy, 
      "\nfor enstrophy\n", skewness_enstrophy, kurtosis_enstrophy)

# Question 2.5
d2u_dx2 = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (2 * dx**2)
d2u_dy2 = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (2 * dx**2)
N_lin = u * du_dx + v * dv_dy
Visc = nu * (d2u_dx2 + d2u_dy2)
for i in range(L_x):
    for j in range(L_y):
        if Visc[i,j] == 0:
            local_Re[i,j] = 0
        else:
            local_Re[i,j] = np.abs(N_lin[i,j] / Visc[i,j])

# Plot of Normalized Kinetic energy
plt.figure(figsize=(8, 6))
plt.pcolor(normalized_kinetic_energy[:: skip, :: skip])
plt.colorbar(label='Normalized Kinetic Energy')
plt.title('Kinetic Energy Field')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Plot of Normalized Vorticity
plt.figure(figsize=(8, 6))
plt.pcolor(normalized_omega_z[:: skip, :: skip])
plt.colorbar(label='Vorticity')
plt.title('Vorticity')
plt.xlabel('Y-axis')
plt.ylabel('X-axis')
plt.show()

# Plot of PDF and CDF
plt.figure(figsize=(12, 6))
# PDF of u, v, w
plt.subplot(1, 2, 1)
plt.plot(bins_u, pdf_u, label='u-component', color='red')
plt.plot(bins_v, pdf_v, label='v-component', color='blue')
plt.plot(bins_w, pdf_w, label='w-component', color='green')
plt.plot(x_values, normal_pdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('PDF of Velocity Components')
plt.yscale('log')
plt.legend()

# CDF of u, v, w
plt.subplot(1, 2, 2)
plt.plot(bins_u, cdf_u, label='u-component', color='red')
plt.plot(bins_v, cdf_v, label='v-component', color='blue')
plt.plot(bins_w, cdf_w, label='w-component', color='green')
plt.plot(x_values, normal_cdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('CDF of Velocity Components')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
# PDF of vorticity
plt.subplot(1, 2, 1)
plt.plot(bins_omega_z, pdf_omega_z, label='Vorticity PDF', color='red')
plt.plot(x_values, normal_pdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('PDF of Vorticity')
plt.yscale('log')
plt.legend()

# CDF of vorticity
plt.subplot(1, 2, 2)
plt.plot(bins_omega_z, cdf_omega_z, label='CDF of Vorticity', color='red')
plt.plot(x_values, normal_cdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('CDF of vorticity')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
# PDF of velocity gradient
plt.subplot(1, 2, 1)
plt.plot(bins_du_dx, pdf_du_dx, label='du_dx PDF', color='red')
plt.plot(bins_dv_dy, pdf_dv_dy, label='dv_dy PDF', color='blue')
plt.plot(x_values, normal_pdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('PDF of velocity gradient')
plt.yscale('log')
plt.legend()

# CDF of velocity gradient
plt.subplot(1, 2, 2)
plt.plot(bins_du_dx, cdf_du_dx, label='du_dx CDF', color='red')
plt.plot(bins_dv_dy, cdf_dv_dy, label='dv_dy CDF', color='blue')
plt.plot(x_values, normal_cdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('CDF of velocity gradient')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
# PDF of enstrophy
plt.subplot(1, 2, 1)
plt.plot(bins_enstrophy, pdf_enstrophy, label='enstrophy PDF', color='red')
plt.plot(x_values, normal_pdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('PDF of enstrophy')
plt.yscale('log')
plt.legend()

# CDF of enstrophy
plt.subplot(1, 2, 2)
plt.plot(bins_enstrophy, cdf_enstrophy, label='enstrophy CDF', color='red')
plt.plot(x_values, normal_cdf, linestyle=':', label='Normal Distribution', color='black')
plt.title('CDF of enstrophy')
plt.yscale('log')
plt.legend()
plt.show()

# PDF of u velocity for 3 different range
plt.plot(bin_u_r1, pdf_u_r1, label='Range 0-128', color='red')
plt.plot(bin_u_r2, pdf_u_r2, label='Range 129-256', color='blue')
plt.plot(bin_u_r3, pdf_u_r3, label='Range 257-512', color='green')
plt.title('PDF of u velocity for different range')
plt.yscale('log')
plt.legend()
plt.show()

# Plot of N_lin, Visc and local Reynolds number
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.pcolor(N_lin[:: skip, :: skip])
plt.colorbar(label='Non-linear term')
plt.title('Non-linear term')

plt.subplot(1,2,2)
plt.pcolor(Visc[:: skip, :: skip])
plt.colorbar(label='linear term')
plt.title('Viscous term')
plt.show()

plt.pcolor(local_Re[:: skip, :: skip], vmin=0, vmax=500)
plt.colorbar(label='Local Reynolds number')
plt.title('Local Reynolds number')
plt.show()

# Given data
plt.pcolor(u[:: skip, :: skip])
plt.show()

