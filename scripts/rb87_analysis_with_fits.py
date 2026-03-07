import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import k as k_B, hbar, m_u
from scipy.optimize import curve_fit
from scipy.stats import norm
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import sys

# Add src to path - MUST be before importing ultracold modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Physical constants
g = 9.81  # m/s^2, gravitational acceleration
m_Rb87 = 87 * m_u  # kg

# Gaussian fitting function


def gaussian_1d(x, amplitude, center, sigma, offset=0):
    """1D Gaussian function for fitting"""
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2)) + offset


def fit_gaussian_profile(positions, n_bins=100, axis_name='X'):
    """
    Fit a Gaussian to a 1D distribution of positions

    Parameters:
    -----------
    positions : array
        1D array of particle positions
    n_bins : int
        Number of histogram bins
    axis_name : str
        Name of axis for display 

    Returns:
    --------
    dict with fit parameters and statistics
    """
    # Create histogram
    counts, bin_edges = np.histogram(positions, bins=n_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Normalize to get optical density-like profile
    optical_density = counts / (np.sum(counts) * bin_width)

    # Initial guess for parameters
    amplitude_guess = np.max(optical_density)
    center_guess = np.mean(positions)
    sigma_guess = np.std(positions)
    offset_guess = np.min(optical_density)

    p0 = [amplitude_guess, center_guess, sigma_guess, offset_guess]

    try:
        # Fit Gaussian
        popt, pcov = curve_fit(gaussian_1d, bin_centers, optical_density,
                               p0=p0, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))

        # Extract parameters
        amplitude, center, sigma, offset = popt
        amp_err, center_err, sigma_err, offset_err = perr

        # Calculate R-squared
        residuals = optical_density - gaussian_1d(bin_centers, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((optical_density - np.mean(optical_density))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate FWHM
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        fwhm_err = 2 * np.sqrt(2 * np.log(2)) * sigma_err

        fit_success = True

    except Exception as e:
        print(f"Warning: Gaussian fit failed for {axis_name} axis: {e}")
        amplitude, center, sigma, offset = np.nan, np.nan, np.nan, np.nan
        amp_err, center_err, sigma_err, offset_err = np.nan, np.nan, np.nan, np.nan
        r_squared = np.nan
        fwhm, fwhm_err = np.nan, np.nan
        popt = None
        fit_success = False

    return {
        'bin_centers': bin_centers,
        'optical_density': optical_density,
        'fit_params': popt,
        'amplitude': amplitude,
        'amplitude_err': amp_err,
        'center': center,
        'center_err': center_err,
        'sigma': sigma,
        'sigma_err': sigma_err,
        'offset': offset,
        'offset_err': offset_err,
        'fwhm': fwhm,
        'fwhm_err': fwhm_err,
        'r_squared': r_squared,
        'n_particles': len(positions),
        'fit_success': fit_success
    }


def analyze_cloud_at_time(trajectory, time_idx, time_ms, n_bins=100):
    """
    Analyze cloud distribution at a specific time

    Returns fitted parameters for all three axes
    """
    positions = trajectory[time_idx]

    # Fit each axis
    fit_x = fit_gaussian_profile(positions[:, 0], n_bins, 'X')
    fit_y = fit_gaussian_profile(positions[:, 1], n_bins, 'Y')
    fit_z = fit_gaussian_profile(positions[:, 2], n_bins, 'Z')

    # Calculate cloud temperature from expansion
    # T = m * sigma_v^2 / k_B, where sigma_v ≈ sigma_expansion / t for ballistic expansion
    if time_ms > 0:
        # Velocity spread from spatial expansion
        v_spread_x = fit_x['sigma'] / (time_ms * 1e-3)  # m/s
        v_spread_y = fit_y['sigma'] / (time_ms * 1e-3)
        v_spread_z = fit_z['sigma'] / (time_ms * 1e-3)

        # Temperature from velocity spread
        T_x = m_Rb87 * v_spread_x**2 / k_B
        T_y = m_Rb87 * v_spread_y**2 / k_B
        T_z = m_Rb87 * v_spread_z**2 / k_B
        T_avg = (T_x + T_y + T_z) / 3
    else:
        T_x = T_y = T_z = T_avg = np.nan

    return {
        'time_ms': time_ms,
        'fit_x': fit_x,
        'fit_y': fit_y,
        'fit_z': fit_z,
        'T_x_nK': T_x * 1e9,
        'T_y_nK': T_y * 1e9,
        'T_z_nK': T_z * 1e9,
        'T_avg_nK': T_avg * 1e9
    }

# %%


def setup_output_directories(output_base: Path) -> dict:
    """Create output directories and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dirs = {
        'base': output_base,
        'figures': output_base / 'figures' / f'f_f_Cloude2_{timestamp}',
        'data': output_base / 'simulations' / f'f_f_Cloude2_{timestamp}',
        'tables': output_base / 'tables' / f'f_f_Cloude2_{timestamp}',
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs
# %%
    """Main simulation routine."""

    # Parse command line arguments
parser = argparse.ArgumentParser(
    description='Simulate free fall of ultracold Rb-87 atomic cloud',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--config', type=str,
    default=str(PROJECT_ROOT / 'configs' / 'experiment_params.yaml'),
    help='Path to configuration file'
)
parser.add_argument(
    '--t-max', type=float, default=25.0,
    help='Maximum simulation time in ms'
)
parser.add_argument(
    '--n-points', type=int, default=200,
    help='Number of time points'
)
parser.add_argument(
    '--single-state', action='store_true',
    help='Use single state cloud instead of mixed state'
)
parser.add_argument(
    '--state', type=str, choices=['mF1', 'mF2'], default='mF2',
    help='State to use for single-state simulation'
)
parser.add_argument(
    '--no-plots', action='store_true',
    help='Skip generating plots'
)
parser.add_argument(
    '--no-save', action='store_true',
    help='Skip saving results'
)
parser.add_argument(
    '--output-dir', type=str, default=None,
    help='Output directory (default: output/simulations)'
)

args = parser.parse_args()

if not args.no_save:
    output_base = Path(
        args.output_dir) if args.output_dir else PROJECT_ROOT / 'output'
    dirs = setup_output_directories(output_base)
    print(f"Output directory: {dirs['data']}")
# %%
# ============================================================================
# SIMULATION SETUP
# ============================================================================


# Experimental parameters from the paper
n_particles = int(1.2e4)  # cambiar e4 por e5
T = 295e-9  # K (295 nK)
omega_l = 2 * np.pi * 45  # rad/s, longitudinal
omega_r = 2 * np.pi * 247  # rad/s, radial

# Calculate parameters
v_thermal = np.sqrt(k_B * T / m_Rb87)
sigma_z = np.sqrt(k_B * T / (m_Rb87 * omega_l**2))
sigma_r = np.sqrt(k_B * T / (m_Rb87 * omega_r**2))

print(f"Thermal velocity (RMS): {v_thermal*100:.4f} cm/s")
print(f"Initial cloud size (longitudinal): {sigma_z*1e6:.2f} μm")
print(f"Initial cloud size (radial): {sigma_r*1e6:.2f} μm")

# Initial position
z0 = 0
x0 = -140e-6  # m
y0 = 0

# Simulation parameters
t_total = 20e-3
dt = 0.1e-3
times = np.arange(0, t_total + dt, dt)

# Use subset for visualization
n_particles_sim = min(10000, n_particles)
print(f"\nSimulating {n_particles_sim} particles")

# Initialize positions
positions = np.zeros((n_particles_sim, 3))
positions[:, 0] = np.random.normal(x0, sigma_r, n_particles_sim)
positions[:, 1] = np.random.normal(y0, sigma_r, n_particles_sim)
positions[:, 2] = np.random.normal(z0, sigma_z, n_particles_sim)

# Initialize velocities
velocities = np.random.normal(0, v_thermal, (n_particles_sim, 3))

# Store trajectory
trajectory = np.zeros((len(times), n_particles_sim, 3))
trajectory[0] = positions.copy()

print(f"\nRunning simulation...")
# Simulate free fall
for i, t in enumerate(times[1:], start=1):
    velocities[:, 0] += g * dt
    positions += velocities * dt
    trajectory[i] = positions.copy()

# ============================================================================
# ANALYSIS WITH GAUSSIAN FITTING
# ============================================================================

# Analysis times from paper
analysis_times = [0, 5, 7, 10, 13, 15, 18, 20]  # ms
analysis_indices = [np.argmin(np.abs(times - t*1e-3)) for t in analysis_times]

print(f"\nAnalyzing cloud profiles and fitting Gaussians...")

# Perform analysis at each time
analysis_results = []
for idx, t_ms in zip(analysis_indices, analysis_times):
    result = analyze_cloud_at_time(trajectory, idx, t_ms, n_bins=80)
    analysis_results.append(result)
    print(f"  Analyzed t = {t_ms} ms")

# ============================================================================
# CREATE COMPREHENSIVE PLOTS WITH FITS
# ============================================================================

print(f"\nGenerating profile plots with Gaussian fits...")

# Plot 4 selected times with detailed fits
plot_times_idx = [0, 2, 4, 6]  # Indices in analysis_results: 0, 7, 13, 18 ms
fig, axes = plt.subplots(4, 3, figsize=(18, 16))

for row, res_idx in enumerate(plot_times_idx):
    result = analysis_results[res_idx]
    t_ms = result['time_ms']

    # X axis
    ax = axes[row, 0]
    fit_x = result['fit_x']
    ax.bar(fit_x['bin_centers']*1e6, fit_x['optical_density'],
           width=(fit_x['bin_centers'][1]-fit_x['bin_centers'][0])*1e6,
           alpha=0.6, color='blue', label='Data')
    if fit_x['fit_success']:
        x_fit = np.linspace(fit_x['bin_centers'].min(),
                            fit_x['bin_centers'].max(), 200)
        y_fit = gaussian_1d(x_fit, *fit_x['fit_params'])
        ax.plot(x_fit*1e6, y_fit, 'r-', linewidth=2, label='Gaussian Fit')
        ax.text(0.05, 0.95, f"$\\mu$ = {fit_x['center']*1e6:.2f} ± {fit_x['center_err']*1e6:.2f} μm\n"
                f"$\\sigma$ = {
                    fit_x['sigma']*1e6:.2f} ± {fit_x['sigma_err']*1e6:.2f} μm\n"
                f"FWHM = {fit_x['fwhm']*1e6:.2f} μm\n"
                f"$R^2$ = {fit_x['r_squared']:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('X Position (μm)', fontsize=10)
    ax.set_ylabel('Optical Density (a.u.)', fontsize=10)
    ax.set_title(f't = {t_ms} ms - X Profile', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Y axis
    ax = axes[row, 1]
    fit_y = result['fit_y']
    ax.bar(fit_y['bin_centers']*1e6, fit_y['optical_density'],
           width=(fit_y['bin_centers'][1]-fit_y['bin_centers'][0])*1e6,
           alpha=0.6, color='green', label='Data')
    if fit_y['fit_success']:
        x_fit = np.linspace(fit_y['bin_centers'].min(),
                            fit_y['bin_centers'].max(), 200)
        y_fit = gaussian_1d(x_fit, *fit_y['fit_params'])
        ax.plot(x_fit*1e6, y_fit, 'r-', linewidth=2, label='Gaussian Fit')
        ax.text(0.05, 0.95, f"$\\mu$ = {fit_y['center']*1e6:.2f} ± {fit_y['center_err']*1e6:.2f} μm\n"
                f"$\\sigma$ = {
                    fit_y['sigma']*1e6:.2f} ± {fit_y['sigma_err']*1e6:.2f} μm\n"
                f"FWHM = {fit_y['fwhm']*1e6:.2f} μm\n"
                f"$R^2$ = {fit_y['r_squared']:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Y Position (μm)', fontsize=10)
    ax.set_ylabel('Optical Density (a.u.)', fontsize=10)
    ax.set_title(f't = {t_ms} ms - Y Profile', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Z axis
    ax = axes[row, 2]
    fit_z = result['fit_z']
    ax.bar(fit_z['bin_centers']*1e6, fit_z['optical_density'],
           width=(fit_z['bin_centers'][1]-fit_z['bin_centers'][0])*1e6,
           alpha=0.6, color='red', label='Data')
    if fit_z['fit_success']:
        x_fit = np.linspace(fit_z['bin_centers'].min(),
                            fit_z['bin_centers'].max(), 200)
        y_fit = gaussian_1d(x_fit, *fit_z['fit_params'])
        ax.plot(x_fit*1e6, y_fit, 'r-', linewidth=2, label='Gaussian Fit')
        ax.text(0.05, 0.95, f"$\\mu$ = {fit_z['center']*1e6:.2f} ± {fit_z['center_err']*1e6:.2f} μm\n"
                f"$\\sigma$ = {
                    fit_z['sigma']*1e6:.2f} ± {fit_z['sigma_err']*1e6:.2f} μm\n"
                f"FWHM = {fit_z['fwhm']*1e6:.2f} μm\n"
                f"$R^2$ = {fit_z['r_squared']:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel('Z Position (μm)', fontsize=10)
    ax.set_ylabel('Optical Density (a.u.)', fontsize=10)
    ax.set_title(f't = {t_ms} ms - Z Profile', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Gaussian Fits of Cloud Density Profiles - $^{87}$Rb Atom Cloud',
             fontsize=14, fontweight='bold')
plt.tight_layout()
dir_fit_figs = dirs['figures'] / 'rb87_gaussian_fits_profiles.png'
plt.savefig(dir_fit_figs, dpi=300, bbox_inches='tight')
print("Saved Gaussian fit profiles")

# ============================================================================
# CREATE DATA TABLES
# ============================================================================

print(f"\nCreating data tables...")

# Table 1: Cloud center positions
table1_data = []
for result in analysis_results:
    row = {
        'Time (ms)': result['time_ms'],
        'X_center (μm)': result['fit_x']['center'] * 1e6,
        'X_center_err (μm)': result['fit_x']['center_err'] * 1e6,
        'Y_center (μm)': result['fit_y']['center'] * 1e6,
        'Y_center_err (μm)': result['fit_y']['center_err'] * 1e6,
        'Z_center (μm)': result['fit_z']['center'] * 1e6,
        'Z_center_err (μm)': result['fit_z']['center_err'] * 1e6,
    }
    table1_data.append(row)

df_centers = pd.DataFrame(table1_data)

# Table 2: Cloud widths (sigma)
table2_data = []
for result in analysis_results:
    row = {
        'Time (ms)': result['time_ms'],
        'σ_X (μm)': result['fit_x']['sigma'] * 1e6,
        'σ_X_err (μm)': result['fit_x']['sigma_err'] * 1e6,
        'σ_Y (μm)': result['fit_y']['sigma'] * 1e6,
        'σ_Y_err (μm)': result['fit_y']['sigma_err'] * 1e6,
        'σ_Z (μm)': result['fit_z']['sigma'] * 1e6,
        'σ_Z_err (μm)': result['fit_z']['sigma_err'] * 1e6,
    }
    table2_data.append(row)

df_widths = pd.DataFrame(table2_data)

# Table 3: FWHM values
table3_data = []
for result in analysis_results:
    row = {
        'Time (ms)': result['time_ms'],
        'FWHM_X (μm)': result['fit_x']['fwhm'] * 1e6,
        'FWHM_X_err (μm)': result['fit_x']['fwhm_err'] * 1e6,
        'FWHM_Y (μm)': result['fit_y']['fwhm'] * 1e6,
        'FWHM_Y_err (μm)': result['fit_y']['fwhm_err'] * 1e6,
        'FWHM_Z (μm)': result['fit_z']['fwhm'] * 1e6,
        'FWHM_Z_err (μm)': result['fit_z']['fwhm_err'] * 1e6,
    }
    table3_data.append(row)

df_fwhm = pd.DataFrame(table3_data)

# Table 4: Fit quality metrics
table4_data = []
for result in analysis_results:
    row = {
        'Time (ms)': result['time_ms'],
        'R²_X': result['fit_x']['r_squared'],
        'R²_Y': result['fit_y']['r_squared'],
        'R²_Z': result['fit_z']['r_squared'],
        'T_avg (nK)': result['T_avg_nK'] if not np.isnan(result['T_avg_nK']) else 0,
    }
    table4_data.append(row)

df_quality = pd.DataFrame(table4_data)

# Table 5: Comprehensive summary
table5_data = []
for result in analysis_results:
    row = {
        'Time (ms)': result['time_ms'],
        'X_pos (μm)': f"{result['fit_x']['center']*1e6:.2f} ± {result['fit_x']['center_err']*1e6:.2f}",
        'Y_pos (μm)': f"{result['fit_y']['center']*1e6:.2f} ± {result['fit_y']['center_err']*1e6:.2f}",
        'Z_pos (μm)': f"{result['fit_z']['center']*1e6:.2f} ± {result['fit_z']['center_err']*1e6:.2f}",
        'σ_X (μm)': f"{result['fit_x']['sigma']*1e6:.2f} ± {result['fit_x']['sigma_err']*1e6:.2f}",
        'σ_Y (μm)': f"{result['fit_y']['sigma']*1e6:.2f} ± {result['fit_y']['sigma_err']*1e6:.2f}",
        'σ_Z (μm)': f"{result['fit_z']['sigma']*1e6:.2f} ± {result['fit_z']['sigma_err']*1e6:.2f}",
    }
    table5_data.append(row)

df_summary = pd.DataFrame(table5_data)

# %%

# Save tables to CSV
dir_cloud_centers = dirs['tables'] / 'cloud_centers.csv'
dir_cloud_widths = dirs['tables'] / 'cloud_widths.csv'
dir_cloud_fwhm = dirs['tables'] / 'cloud_fwhm.csv'
dir_cloud_quality = dirs['tables'] / 'cloud_fit_quality.csv'
dir_cloud_summary = dirs['tables'] / 'cloud_summary.csv'

df_centers.to_csv(dir_cloud_centers, index=False, float_format='%.4f')
df_widths.to_csv(dir_cloud_widths, index=False, float_format='%.4f')
df_fwhm.to_csv(dir_cloud_fwhm, index=False, float_format='%.4f')
df_quality.to_csv(dir_cloud_quality, index=False, float_format='%.4f')
df_summary.to_csv(dir_cloud_summary, index=False)

print("Saved CSV tables")

# %% Print tables to console
print("\n" + "="*80)
print("TABLE 1: Cloud Center Positions (from Gaussian Fits)")
print("="*80)
print(df_centers.to_string(index=False))

print("\n" + "="*80)
print("TABLE 2: Cloud Widths (σ from Gaussian Fits)")
print("="*80)
print(df_widths.to_string(index=False))

print("\n" + "="*80)
print("TABLE 3: Cloud FWHM")
print("="*80)
print(df_fwhm.to_string(index=False))

print("\n" + "="*80)
print("TABLE 4: Fit Quality and Derived Temperature")
print("="*80)
print(df_quality.to_string(index=False))

# ============================================================================
# CREATE PLOTS OF FITTED PARAMETERS VS TIME
# ============================================================================

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

# Extract data for plotting
times_ms = [r['time_ms'] for r in analysis_results]
x_centers = [r['fit_x']['center']*1e6 for r in analysis_results]
y_centers = [r['fit_y']['center']*1e6 for r in analysis_results]
z_centers = [r['fit_z']['center']*1e6 for r in analysis_results]
x_centers_err = [r['fit_x']['center_err']*1e6 for r in analysis_results]
y_centers_err = [r['fit_y']['center_err']*1e6 for r in analysis_results]
z_centers_err = [r['fit_z']['center_err']*1e6 for r in analysis_results]

x_sigmas = [r['fit_x']['sigma']*1e6 for r in analysis_results]
y_sigmas = [r['fit_y']['sigma']*1e6 for r in analysis_results]
z_sigmas = [r['fit_z']['sigma']*1e6 for r in analysis_results]
x_sigmas_err = [r['fit_x']['sigma_err']*1e6 for r in analysis_results]
y_sigmas_err = [r['fit_y']['sigma_err']*1e6 for r in analysis_results]
z_sigmas_err = [r['fit_z']['sigma_err']*1e6 for r in analysis_results]

# Plot 1: Center positions vs time
axes[0, 0].errorbar(times_ms, x_centers, yerr=x_centers_err, fmt='o-',
                    label='X (gravity)', linewidth=2, markersize=6, capsize=4)
axes[0, 0].errorbar(times_ms, y_centers, yerr=y_centers_err, fmt='s-',
                    label='Y', linewidth=2, markersize=6, capsize=4)
axes[0, 0].errorbar(times_ms, z_centers, yerr=z_centers_err, fmt='^-',
                    label='Z (beam)', linewidth=2, markersize=6, capsize=4)
axes[0, 0].set_xlabel('Time (ms)', fontsize=11)
axes[0, 0].set_ylabel('Cloud Center Position (μm)', fontsize=11)
axes[0, 0].set_title('Cloud Center Evolution', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cloud widths vs time
axes[0, 1].errorbar(times_ms, x_sigmas, yerr=x_sigmas_err, fmt='o-',
                    label='σ_X', linewidth=2, markersize=6, capsize=4)
axes[0, 1].errorbar(times_ms, y_sigmas, yerr=y_sigmas_err, fmt='s-',
                    label='σ_Y', linewidth=2, markersize=6, capsize=4)
axes[0, 1].errorbar(times_ms, z_sigmas, yerr=z_sigmas_err, fmt='^-',
                    label='σ_Z', linewidth=2, markersize=6, capsize=4)
axes[0, 1].set_xlabel('Time (ms)', fontsize=11)
axes[0, 1].set_ylabel('Cloud Width σ (μm)', fontsize=11)
axes[0, 1].set_title('Cloud Size Evolution', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: R-squared values
r2_x = [r['fit_x']['r_squared'] for r in analysis_results]
r2_y = [r['fit_y']['r_squared'] for r in analysis_results]
r2_z = [r['fit_z']['r_squared'] for r in analysis_results]
axes[0, 2].plot(times_ms, r2_x, 'o-', label='R²_X', linewidth=2, markersize=6)
axes[0, 2].plot(times_ms, r2_y, 's-', label='R²_Y', linewidth=2, markersize=6)
axes[0, 2].plot(times_ms, r2_z, '^-', label='R²_Z', linewidth=2, markersize=6)
axes[0, 2].axhline(y=0.99, color='r', linestyle='--',
                   alpha=0.5, label='R² = 0.99')
axes[0, 2].set_xlabel('Time (ms)', fontsize=11)
axes[0, 2].set_ylabel('R² (Fit Quality)', fontsize=11)
axes[0, 2].set_title('Gaussian Fit Quality', fontsize=12, fontweight='bold')
axes[0, 2].set_ylim([0.95, 1.005])
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: X displacement (fall distance)
x_displacement = np.array(x_centers) - x_centers[0]
axes[1, 0].plot(times_ms, x_displacement, 'bo-',
                label='Measured', linewidth=2, markersize=6)
# Theoretical prediction
t_array = np.array(times_ms) * 1e-3
theoretical = 0.5 * g * t_array**2 * 1e6  # Convert to μm
axes[1, 0].plot(times_ms, theoretical, 'r--',
                label='Theory: ½gt²', linewidth=2)
axes[1, 0].set_xlabel('Time (ms)', fontsize=11)
axes[1, 0].set_ylabel('Fall Distance (μm)', fontsize=11)
axes[1, 0].set_title('Cloud Fall (X Direction)',
                     fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Aspect ratios
aspect_xy = np.array(x_sigmas) / np.array(y_sigmas)
aspect_xz = np.array(x_sigmas) / np.array(z_sigmas)
aspect_yz = np.array(y_sigmas) / np.array(z_sigmas)
axes[1, 1].plot(times_ms, aspect_xy, 'o-',
                label='σ_X/σ_Y', linewidth=2, markersize=6)
axes[1, 1].plot(times_ms, aspect_xz, 's-',
                label='σ_X/σ_Z', linewidth=2, markersize=6)
axes[1, 1].plot(times_ms, aspect_yz, '^-',
                label='σ_Y/σ_Z', linewidth=2, markersize=6)
axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_xlabel('Time (ms)', fontsize=11)
axes[1, 1].set_ylabel('Aspect Ratio', fontsize=11)
axes[1, 1].set_title('Cloud Shape Evolution', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Derived temperature
T_avg = [r['T_avg_nK'] for r in analysis_results[1:]]  # Skip t=0
times_nonzero = times_ms[1:]
axes[1, 2].plot(times_nonzero, T_avg, 'mo-', linewidth=2, markersize=6)
axes[1, 2].axhline(y=T*1e9, color='g', linestyle='--', linewidth=2,
                   label=f'Input T = {T*1e9:.0f} nK')
axes[1, 2].set_xlabel('Time (ms)', fontsize=11)
axes[1, 2].set_ylabel('Temperature (nK)', fontsize=11)
axes[1, 2].set_title('Temperature from Cloud Expansion',
                     fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Fitted Cloud Parameters vs Time - $^{87}$Rb Atom Cloud',
             fontsize=14, fontweight='bold')
plt.tight_layout()
dir_fit_figs_evol = dirs['figures'] / 'rb87_fitted_parameters_evolution.png'
plt.savefig(dir_fit_figs_evol,
            dpi=300, bbox_inches='tight')
print("Saved parameter evolution plots")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)
print(f"\nInitial Parameters (t = 0 ms):")
print(f"  Input temperature: {T*1e9:.1f} nK")
print(f"  Fitted σ_X: {analysis_results[0]['fit_x']['sigma']*1e6:.2f} ± "
      f"{analysis_results[0]['fit_x']['sigma_err']*1e6:.2f} μm")
print(f"  Fitted σ_Y: {analysis_results[0]['fit_y']['sigma']*1e6:.2f} ± "
      f"{analysis_results[0]['fit_y']['sigma_err']*1e6:.2f} μm")
print(f"  Fitted σ_Z: {analysis_results[0]['fit_z']['sigma']*1e6:.2f} ± "
      f"{analysis_results[0]['fit_z']['sigma_err']*1e6:.2f} μm")
print(f"  Expected σ_radial: {sigma_r*1e6:.2f} μm")
print(f"  Expected σ_longitudinal: {sigma_z*1e6:.2f} μm")

print(f"\nFinal Parameters (t = {analysis_results[-1]['time_ms']:.0f} ms):")
print(f"  Fitted σ_X: {analysis_results[-1]['fit_x']['sigma']*1e6:.2f} ± "
      f"{analysis_results[-1]['fit_x']['sigma_err']*1e6:.2f} μm")
print(f"  Fitted σ_Y: {analysis_results[-1]['fit_y']['sigma']*1e6:.2f} ± "
      f"{analysis_results[-1]['fit_y']['sigma_err']*1e6:.2f} μm")
print(f"  Fitted σ_Z: {analysis_results[-1]['fit_z']['sigma']*1e6:.2f} ± "
      f"{analysis_results[-1]['fit_z']['sigma_err']*1e6:.2f} μm")
print(f"  Fall distance: {x_displacement[-1]:.1f} μm")
print(f"  Theoretical fall: {theoretical[-1]:.1f} μm")
print(f"  Derived temperature: {analysis_results[-1]['T_avg_nK']:.1f} nK")

print("\n" + "="*80)
print("All files saved successfully!")
print("="*80)

# %%
dirs.keys()
figsdir = dirs['figures']
# %%
figsdir
