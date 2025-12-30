import numpy as np
import matplotlib as mpl
import os
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import linregress

def load_or_compute(filename, compute_func, *args):
    if os.path.exists(filename):
        print(f"Loading {filename}...")
        data = np.load(filename)
        return {key: data[key] for key in data.files}
    else:
        print(f"Computing and saving {filename}...")
        result = compute_func(*args)
        np.savez_compressed(filename, **result)
        return result

# Load data
data = np.load("isotropic1024_stack3.npz" )
u = data["u"]
v = data["v"]
w = data["w"]

u1 = u[:, :, 1]
v1 = v[:, :, 1]
w1 = w[:, :, 1]

# Grid parameters
L = 2 * np.pi
N = 1024
dx = L/N
dy = L/N
dz = L/N

# Kolmogorov scales
nu = 0.000185 
u_rms = np.sqrt(np.mean(u1**2 + v1**2))
epsilon = u_rms**3 / 1.364
eta = (nu**3 / epsilon)**(1/4)
tau_eta = (nu / epsilon)**(1/2)

# Control parameters
Np = 10000
time = 20
dt_eta = tau_eta / 10

# Particle parameters
epsilon_perturbation = np.array([0.1, 0.5, 1, 5, 10]) * dx
t_plot = np.arange(0, time+dt_eta, dt_eta)

x_a = np.random.uniform(0, L, Np)
y_a = np.random.uniform(0, L, Np)
theta = np.random.uniform(0, 2*np.pi, Np)

x = np.linspace(0, 2*np.pi, N, endpoint=False)
y = np.linspace(0, 2*np.pi, N, endpoint=False)
u_interpolated = RegularGridInterpolator((x, y), u1, method='linear', bounds_error=False, fill_value=None)
v_interpolated = RegularGridInterpolator((x, y), v1, method='linear', bounds_error=False, fill_value=None)

# Partical tracker function
def tracker(time, i, x_old, y_old):
    T = 0
    x_values = []
    y_values = []
    x_old = x_old[i]
    y_old = y_old[i]
    x = x_old
    y = y_old
    x_values.append(x_old)
    y_values.append(y_old)
    
    while T < time:
        if x_old > L or x_old < 0:
            x = (x_old + L) % L

        if y_old > L or y_old < 0:
            y = (y_old + L) % L
  
        position = np.array([x, y])
        
        u_position = u_interpolated(position)
        v_position = v_interpolated(position)
        
        x_new = x_old + u_position[0] * dt_eta
        y_new = y_old + v_position[0] * dt_eta
        
        x_values.append(x_new)
        y_values.append(y_new)

        x_old = x_new
        y_old = y_new
        x = x_new
        y = y_new
        
        T += dt_eta
        
    return x_values, y_values

def compute_particle_tracks(x_a, y_a, time, Np):
    x_data_list_a = []
    y_data_list_a = []
    for points in range(len(x_a)):
        x_plot, y_plot = tracker(time, points, x_a, y_a)
        x_data_list_a.append(x_plot)
        y_data_list_a.append(y_plot)
        print(f"Set A, Particle: {points}")
    return {"x_data_a": np.array(x_data_list_a), "y_data_a": np.array(y_data_list_a)}

# Load or compute Set A trajectories
track_data = load_or_compute("particle_tracks.npz", compute_particle_tracks, x_a, y_a, time, Np)
x_data_a = track_data["x_data_a"]
y_data_a = track_data["y_data_a"]

# Perturbation calculations
def compute_perturbations(x_a, y_a, theta, epsilon_perturbation, time, Np, x_data_a, y_data_a):
    x_data_all = []
    y_data_all = []
    r_square_perturbation = []
    
    for i in range(len(epsilon_perturbation)):
        x_perturbation = x_a + (epsilon_perturbation[i] * np.cos(theta[i]))
        y_perturbation = y_a + (epsilon_perturbation[i] * np.sin(theta[i]))
        x_data_list_perturbed = []
        y_data_list_perturbed = []
        for points in range(len(x_perturbation)):
            x_plot, y_plot = tracker(time, points, x_perturbation, y_perturbation)
            x_data_list_perturbed.append(x_plot)
            y_data_list_perturbed.append(y_plot)
            print(f"Epsilon: {epsilon_perturbation[i] / dx}, Particle: {points}")

        x_data_perturbed = np.array(x_data_list_perturbed)
        y_data_perturbed = np.array(y_data_list_perturbed)
        
        r_square_data = []
        for j in range(Np):
            r_square = np.zeros(len(t_plot))
            for k in range(len(t_plot)):
                r_square[k] = (x_data_a[j, k] - x_data_perturbed[j, k])**2 + (y_data_a[j, k] - y_data_perturbed[j, k])**2
            r_square_data.append(r_square)
        r_square_mean = np.mean(r_square_data, axis=0)
        r_square_perturbation.append(r_square_mean)
        x_data_all.append(x_data_perturbed)
        y_data_all.append(y_data_perturbed)
    
    return {
        "x_data_all": np.array(x_data_all),
        "y_data_all": np.array(y_data_all),
        "r_square_perturbation": np.array(r_square_perturbation)
    }

# Load or compute perturbation data
pert_data = load_or_compute("perturbation_data.npz", 
                           compute_perturbations,
                           x_a, y_a, theta, epsilon_perturbation, time, Np, x_data_a, y_data_a)
x_data_all = pert_data["x_data_all"]
y_data_all = pert_data["y_data_all"]
r_square_perturbation = pert_data["r_square_perturbation"]

# Part 1: Coherent Structures and Flow Topology
print("Part 1: Coherent Structures and Flow Topology")

# (Q.1.2): PDF of eigenvalues
# Velocity gradient tensor calculation
du_dx = (np.roll(u1, -1, axis=0) - np.roll(u1, 1, axis=0)) / (2 * dx)
du_dy = (np.roll(u1, -1, axis=1) - np.roll(u1, 1, axis=1)) / (2 * dy)
du_dz = (u[:, :, 2] - u[:, :, 0]) / (2 * dz)

dv_dx = (np.roll(v1, -1, axis=0) - np.roll(v1, 1, axis=0)) / (2 * dx)
dv_dy = (np.roll(v1, -1, axis=1) - np.roll(v1, 1, axis=1)) / (2 * dy)
dv_dz = (v[:, :, 2] - v[:, :, 0]) / (2 * dz)

dw_dx = (np.roll(w1, -1, axis=0) - np.roll(w1, 1, axis=0)) / (2 * dx)
dw_dy = (np.roll(w1, -1, axis=1) - np.roll(w1, 1, axis=1)) / (2 * dy)
dw_dz = (w[:, :, 2] - w[:, :, 0]) / (2 * dz)

# Eigenvalue calculations
def compute_eigenvalues(du_dx, dv_dy, dw_dz, du_dy, dv_dx, du_dz, dw_dx, dv_dz, dw_dy, N):
    eigenvalues = np.empty((N, N, 3), dtype=np.complex128)
    P = -(du_dx + dv_dy + dw_dz)
    Q = (dv_dy * dw_dz) - (dv_dz * dw_dy) + (du_dx * dw_dz) - (du_dz * dw_dx) + (du_dx * dv_dy) - (du_dy * dv_dx)
    R = -(du_dx * ((dv_dy * dw_dz) - (dv_dz * dw_dy)) - du_dy * ((dv_dx * dw_dz) - (dv_dz * dw_dx)) + du_dz * ((dv_dx * dw_dy) - (dv_dy * dw_dx)))
    
    for i in range(N):
        for j in range(N):
            coeffs = [1, P[i, j], Q[i, j], R[i, j]]
            eigenvalues[i, j] = np.roots(coeffs)
        print(f"Eigenvalue calculation at grid {i}")
    
    return {"eigenvalues": eigenvalues, "P": P, "Q": Q, "R": R}

# Load or compute eigenvalues
eigen_data = load_or_compute("eigenvalue_data.npz",
                            compute_eigenvalues,
                            du_dx, dv_dy, dw_dz, du_dy, dv_dx, du_dz, dw_dx, dv_dz, dw_dy, N)
eigenvalues = eigen_data["eigenvalues"]
P = eigen_data["P"]
Q = eigen_data["Q"]
R = eigen_data["R"]

eigen_sorted = np.sort(eigenvalues, axis=2)
lambda3 = np.abs(eigen_sorted[:, :, 0]).flatten()
lambda2 = np.abs(eigen_sorted[:, :, 1]).flatten()
lambda1 = np.abs(eigen_sorted[:, :, 2]).flatten()

# Plot of PDFs 
plt.figure(figsize=(8,6))
for lambdas, label, color in zip([lambda1, lambda2, lambda3], [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$'], ['r', 'g', 'b']):
    counts, bins = np.histogram(lambdas, bins=200, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Mask out very small values
    masked_counts = np.where(counts > 1e-7, counts, np.nan)
    plt.semilogy(bin_centers, masked_counts, label=label, color=color)

plt.xlabel(r'Eigenvalue ($\lambda$)$\rightarrow$')
plt.ylabel(r'PDF (log scale)$\rightarrow$')
plt.xlim(0, 100)
plt.grid(True, which="both")
plt.legend()
plt.title("PDF of Eigenvalues of the Velocity Gradient Tensor")
plt.savefig("eigenvalue_pdf.png", dpi=300)
plt.show()

# (Q.1.3): Proof for $\sum \lambda_i = 0$
print("Sum of eigen values =", np.mean(P))

# (Q.1.4): Q-R plot
Q_rms = np.sqrt(np.mean(Q**2))
R_rms = np.sqrt(np.mean(R**2))

Q_norm = Q / Q_rms
R_norm = R / R_rms

# Plot of Q
skip = 1

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.imshow(Q_norm[::skip, ::skip], cmap='viridis', aspect='auto', vmin=-0.1, vmax=0.1, extent = [0, 6.28, 0, 6.28])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\frac{Q}{Q_{rms}}$')

# Plot of R
plt.subplot(1, 2, 2)
plt.imshow(R_norm[::skip, ::skip], cmap='viridis', aspect='auto', vmin=-0.1, vmax=0.1, extent = [0, 6.28, 0, 6.28])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$\frac{R}{R_{rms}}$')

plt.suptitle('Q and R Fields')
plt.savefig("Q_R_normalized.png", dpi=300)
plt.show()

# (Q.1.5): Q-R scatter plot with discriminant curve
omega_x = dw_dy - dv_dz
omega_y = du_dz - dw_dx
omega_z = dv_dx - du_dy
enstrophy = np.mean(omega_x**2 + omega_y**2 + omega_z**2)
Q_w = enstrophy / 4
Q_scatter = Q / Q_w
R_scatter = R / (Q_w**(3/2))

R_vals = np.linspace(np.min(R_scatter), np.max(R_scatter), 1000)
Q_curve = -((27/4) * R_vals**2)**(1/3)

# Plot of Q-R scatter
skip = 1
plt.figure(figsize=(8,6))
plt.scatter(R_scatter[::skip], Q_scatter[::skip], s=1, color='black', alpha=0.5, label='Data Points')
plt.plot(R_vals, Q_curve, color='red', label='Discriminant Curve')
plt.xlabel(r'R')
plt.ylabel(r'Q')
plt.legend()
plt.grid(True)
plt.xlim(-150, 150)
plt.ylim(-50, 50)
plt.title('Q-R Scatter with Discriminant Curve')
plt.savefig("Q_R_scatter.png", dpi=300)
plt.show()

# Part 2: Lagrangian Aspects of Turbulence
print("Part 2: Lagrangian Aspects of Turbulence")

# (Q.2.1) & (Q.2.2): Lagrangian trajectories
particles = [20, 1000]
for num_particles in particles:
    plt.figure(figsize=(15, 5))
    
    time_step_at_1 = int(1 / dt_eta)
    time_step_at_5 = int(5 / dt_eta)
    time_step_at_10 = int(10 / dt_eta)
    
    plt.subplot(1, 3, 1)
    plt.title(f"Trajectories for Np = {num_particles} at t = 1")
    plt.xlabel(r'x$\rightarrow$')
    plt.ylabel(r'y$\rightarrow$')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.title(f"Trajectories for Np = {num_particles} at t = 5")
    plt.xlabel(r'x$\rightarrow$')
    plt.ylabel(r'y$\rightarrow$')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.title(f"Trajectories for Np = {num_particles} at t = 10")
    plt.xlabel(r'x$\rightarrow$')
    plt.ylabel(r'y$\rightarrow$')
    plt.grid(True)
    
    for particle in range(num_particles):
        plt.subplot(1, 3, 1)
        plt.plot(x_data_a[particle, :time_step_at_1+1], 
                 y_data_a[particle, :time_step_at_1+1], linewidth=0.5)
        
        plt.subplot(1, 3, 2)
        plt.plot(x_data_a[particle, :time_step_at_5+1], 
                 y_data_a[particle, :time_step_at_5+1], linewidth=0.5)
        
        plt.subplot(1, 3, 3)
        plt.plot(x_data_a[particle, :time_step_at_10+1], 
                 y_data_a[particle, :time_step_at_10+1], linewidth=0.5)
            
    plt.tight_layout()
    plt.savefig(f"particle_trajectories_{num_particles}.png", dpi=300)
    plt.show()

# Q.2.3: Mean-Square-Displacement (MSD)
displacement_data = []
diffusive_range = (t_plot >= 10) & (t_plot < 20)
for num_particles in range(10000):
    displacement = np.zeros(len(t_plot))
    displacement = (x_data_a[num_particles, :] - x_data_a[num_particles, 0])**2 + (y_data_a[num_particles, :] - y_data_a[num_particles, 0])**2
    displacement_data.append(displacement)
displacement_plot = np.mean(displacement_data, axis=0)

slope, intercept, _, _, _ = linregress(t_plot[diffusive_range], displacement_plot[diffusive_range])
plt.figure(figsize=(8,6))
plt.loglog(t_plot, displacement_plot, label="MSD curve")
plt.loglog(t_plot[diffusive_range], intercept + slope * t_plot[diffusive_range], label="Diffusive fit")
plt.xlabel(r'Time$\rightarrow$')
plt.ylabel(r'Mean Squared Displacement (MSD)$\rightarrow$')
plt.grid(True, which="both")
plt.legend()
plt.title(r'Mean Squared Displacement (MSD) vs Time (at Np = $10^4$)')
plt.savefig("mean_squared_displacement.png")
plt.show()

# Q.2.4: Calculating the turbulent diffusivity
print("Turbulent diffusivity =", slope)

# Part 3: Richardson pair dispersion
print("Part 3: Richardson pair dispersion")

# (Q.2.5a): Pair trajectories
x_data_all = np.array(x_data_all)
y_data_all = np.array(y_data_all)

skip = 100
plt.figure(figsize=(8,6))
for i in range(20):
    x_plot = x_data_a[i,:time_step_at_10 + 1]
    y_plot = y_data_a[i,:time_step_at_10 + 1]
    set_a = plt.plot(x_plot[::skip], y_plot[::skip], '.', markersize=1, linestyle = 'None', color = 'k', label='Set A' if i == 0 else "") 

for i in range(20):
    x_plot = x_data_all[1, i, :time_step_at_10 + 1]
    y_plot = y_data_all[1, i, :time_step_at_10 + 1]
    set_b = plt.plot(x_plot[::skip], y_plot[::skip], '.', markersize=1, linestyle = 'None', color = 'r', label='Set B' if i == 0 else "")

plt.xlabel(r'x$\rightarrow$')
plt.ylabel(r'y$\rightarrow$')
plt.grid(True)
plt.legend()
plt.title(r'Pair Trajectories(at Np = 20, $\epsilon = 0.5\Delta x$, T = 10)')
plt.savefig("pair_trajectories.png")
plt.show()

# Q.2.5b: pair separation of trajectories on loglog scale
i=0

T_by_tau_eta = t_plot / tau_eta
inertial_range = (T_by_tau_eta > 60) & (T_by_tau_eta < 450)
r_square_perturbation = np.array(r_square_perturbation)
corlors = ['r', 'g', 'b', 'c', 'm']

print("For T^m scaling in the inertial range")
plt.figure(figsize=(8,6))
for eps in epsilon_perturbation:
    r_square_by_epsilon = r_square_perturbation[i, :] / eps**2
    plt.loglog(T_by_tau_eta, r_square_by_epsilon, label=f"ε = {eps/dx:.1f}", color=corlors[i])
    slope, intercept, _, _, _ = linregress(np.log(T_by_tau_eta[inertial_range]), np.log(r_square_by_epsilon[inertial_range]))
    print(f"m for ε = {eps/dx:.1f}: {slope:.4f}")
    plt.loglog(T_by_tau_eta[inertial_range], np.exp(intercept)* T_by_tau_eta[inertial_range]**slope, label=f"Fit for ε = {eps/dx:.1f}", linestyle='--', color=corlors[i])
    i += 1
    
plt.xlabel(r'$t / \tau_\eta \rightarrow$')
plt.ylabel(r'$\Delta r^2 / \epsilon^2 \rightarrow$')
plt.grid(True, which="both")
plt.legend()
plt.title(r'Pair Separation of Trajectories (Log-Log Scale)(at Np = $10^4$)')    
plt.savefig("pair separation trajectories (Log-Log Scale).png", dpi=300)
plt.show()

# Q.2.5c: pair separation of trajectories on Semilog scale
i = 0
print("For Lyapunov exponent Λ")
mask = (T_by_tau_eta >= 0) & (T_by_tau_eta < 120)
plt.figure(figsize=(8,6))
for eps in epsilon_perturbation:
    r_square_by_epsilon = r_square_perturbation[i, :] / eps**2
    plt.semilogy(T_by_tau_eta, r_square_by_epsilon, label=f"ε = {eps/dx:.1f}", color=corlors[i])
    slope, intercept, _, _, _ = linregress(T_by_tau_eta[mask], np.log(r_square_by_epsilon[mask]))
    print(f"Λ for ε = {eps/dx:.1f}: {slope:.4f}")
    plt.semilogy(T_by_tau_eta[mask],np.exp(intercept + slope * T_by_tau_eta[mask]), label=f"Fit for ε = {eps/dx:.1f}", linestyle='--', color=corlors[i])
    i += 1

plt.xlabel(r'$t / \tau_\eta \rightarrow$')
plt.ylabel(r'$\Delta r^2 / \epsilon^2 \rightarrow$')
plt.legend()
plt.grid(True, which="both")
plt.title(r'Pair Separation of Trajectories (Semilog Scale)(at Np = $10^4$)')
plt.savefig("pair_separation_trajectories (Semilog scale).png", dpi=300)
plt.show()