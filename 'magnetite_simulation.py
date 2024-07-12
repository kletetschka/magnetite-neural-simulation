import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy.integrate import odeint

def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def generate_particles(n_particles, min_size, max_size, cleft_size):
    sizes = np.random.uniform(min_size, max_size, n_particles)
    positions = np.random.rand(n_particles, 3) * cleft_size
    return sizes, positions

def calculate_magnetic_moment(size):
    # Magnetite properties
    saturation_magnetization = 480e3  # A/m for magnetite
    volume = (4/3) * np.pi * (size/2)**3  # m^3
    moment = saturation_magnetization * volume  # Am^2
    return moment

def induced_electric_field(positions, moments, target_position):
    r = target_position - positions
    r_mag = np.linalg.norm(r, axis=1)
    r_unit = r / r_mag[:, np.newaxis]
    
    m_dot_r = np.sum(moments * r_unit, axis=1)
    
    E = np.sum(3 * r_unit * m_dot_r[:, np.newaxis] - moments, axis=0) / (4 * np.pi * np.sum(r_mag**3))
    
    debug_print(f"  Max r_mag: {np.max(r_mag):.6e}, Min r_mag: {np.min(r_mag):.6e}")
    debug_print(f"  Max moment: {np.max(np.linalg.norm(moments, axis=1)):.6e}, Min moment: {np.min(np.linalg.norm(moments, axis=1)):.6e}")
    debug_print(f"  Max m_dot_r: {np.max(m_dot_r):.6e}, Min m_dot_r: {np.min(m_dot_r):.6e}")
    debug_print(f"  Induced E-field: {E}")
    
    return E

def hodgkin_huxley(y, t, I_ext):
    V, m, n, h = y
    g_Na, g_K, g_L = 120, 36, 0.3
    E_Na, E_K, E_L = 50, -77, -54.4
    C_m = 1.0

    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = 4.0 * np.exp(-(V + 65) / 18)
    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h = 1.0 / (1 + np.exp(-(V + 35) / 10))
    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = 0.125 * np.exp(-(V + 65) / 80)

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)

    dVdt = (I_ext - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1 - m) - beta_m * m
    dndt = alpha_n * (1 - n) - beta_n * n
    dhdt = alpha_h * (1 - h) - beta_h * h

    return [dVdt, dmdt, dndt, dhdt]

def has_action_potential(solution, threshold=0):
    return np.max(solution[:, 0]) > threshold

def plot_particles(fig, gs, positions, sizes):
    ax = fig.add_subplot(gs[3, 1], projection='3d')
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=sizes*1e9, s=20, alpha=0.6, cmap='viridis')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.set_title('Particle Arrangement')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_zlim(0, 50)
    cbar = plt.colorbar(scatter, ax=ax, label='Particle Size (nm)')
    cbar.set_ticks(np.linspace(19, 24, 6))
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect((10, 10, 1))

    # Add a second view (top view)
    ax2 = fig.add_subplot(gs[3, 2], projection='3d')
    ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                c=sizes*1e9, s=20, alpha=0.6, cmap='viridis')
    ax2.set_xlabel('X (nm)')
    ax2.set_ylabel('Y (nm)')
    ax2.set_zlabel('Z (nm)')
    ax2.set_title('Particle Arrangement (Top View)')
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, 500)
    ax2.set_zlim(0, 50)
    ax2.view_init(elev=90, azim=0)
    ax2.set_box_aspect((10, 10, 1))

# Simulation parameters
n_particles = 200
min_size, max_size = 19e-9, 24e-9  # 19-24 nm diameter range
cleft_size = np.array([500e-9, 500e-9, 50e-9])
scaling_factor = 5e-6
t = np.linspace(0, 50, 1000)
y0 = [-65, 0.05, 0.6, 0.3]

# Set base current
I_base = 5.4941  # μA/cm²

# Run multiple simulations
n_simulations = 10
results = []
last_particles = None

debug_print("Running multiple simulations...")
for i in range(n_simulations):
    debug_print(f"Simulation {i+1}:")
    sizes, positions = generate_particles(n_particles, min_size, max_size, cleft_size)
    moments = np.array([calculate_magnetic_moment(size) * np.random.rand(3) for size in sizes])
    E_induced = induced_electric_field(positions, moments, cleft_size / 2)
    I_induced = np.linalg.norm(E_induced) * scaling_factor
    I_total = I_base + I_induced
    debug_print(f"  I_base: {I_base:.6f}, I_induced: {I_induced:.6e}, I_total: {I_total:.6f}")
    solution = odeint(hodgkin_huxley, y0, t, args=(I_total,))
    results.append((I_total, solution))
    debug_print(f"  Has AP: {has_action_potential(solution)}\n")
    
    if i == n_simulations - 1:
        last_particles = (positions, sizes)

# Calculate summary statistics
I_totals = [result[0] for result in results]
ap_counts = [has_action_potential(result[1]) for result in results]

# Print summary statistics
debug_print("\nSummary Statistics:")
debug_print(f"Mean Total Current: {np.mean(I_totals):.6f}")
debug_print(f"Std Dev Total Current: {np.std(I_totals):.6f}")
debug_print(f"Min Total Current: {np.min(I_totals):.6f}")
debug_print(f"Max Total Current: {np.max(I_totals):.6f}")
debug_print(f"Simulations with APs: {sum(ap_counts)}/{n_simulations}")

# Plotting
fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(5, 3)

for i, (I_total, solution) in enumerate(results):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.plot(t, solution[:, 0])
    ax.set_title(f'Sim {i+1}: I_total = {I_total:.6f}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_ylim(-80, 40)
    ax.grid(True)
    
    if has_action_potential(solution):
        ax.set_facecolor('#e6ffe6')  # Light green background

# Use one subplot for summary statistics
ax_stats = fig.add_subplot(gs[3, 0])
ax_stats.bar(range(1, n_simulations + 1), I_totals)
ax_stats.set_xlabel('Simulation Number')
ax_stats.set_ylabel('Total Current (μA/cm²)')
ax_stats.set_title('Total Currents and AP Occurrences')
ax_stats.set_ylim(min(I_totals) - 0.001, max(I_totals) + 0.001)

for i, (I_total, has_ap) in enumerate(zip(I_totals, ap_counts)):
    color = 'green' if has_ap else 'red'
    ax_stats.text(i+1, I_total, 'AP' if has_ap else 'No AP', ha='center', va='bottom', color=color)

# Add particle arrangement plots
plot_particles(fig, gs, last_particles[0] * 1e9, last_particles[1])  # Convert positions to nm

plt.tight_layout()
plt.savefig('magnetite_nanoparticles_analysis_with_arrangement.png', dpi=300, bbox_inches='tight')
debug_print("Figure saved. Analysis completed.")