# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:41:01 2026

@author: clher
"""
#!/usr/bin/env python3
"""
Simulate free fall of ultracold Rb-87 atomic cloud.

This script:
1. Loads experimental parameters from config file
2. Creates a thermal cloud (single state or mixed state)
3. Simulates free fall evolution
4. Extracts position, velocity, and size vs time
5. Generates profiles at specified observation times
6. Saves results and plots

Usage:
    python simulate_free_fall.py
    python simulate_free_fall.py --config configs/experiment_params.yaml
    python simulate_free_fall.py --t-max 25 --n-points 500
    python simulate_free_fall.py --single-state --state mF2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
from datetime import datetime
from typing import Optional

# Add src to path - MUST be before importing ultracold modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Now import our modules
from ultracold.utils.constants import (
    STATE_F2_MF2, 
    STATE_F2_MF1,
    ExperimentParams,
    G, 
    KB, 
    M_RB87
)

from ultracold.simulation.cloud import ThermalCloud, MixedStateCloud

from ultracold.simulation.free_fall import (
    simulate_free_fall,
    create_experiment_cloud,
    FreeFallSimulation
)

from ultracold.analysis.profiles import (
    extract_horizontal_profile,
    extract_vertical_profile,
    profile_from_simulation,
    Profile1D
)

from ultracold.analysis.cloud_analysis import (
    analyze_cloud_image,
    extract_position_vs_time,
    extract_size_vs_time,
    CloudParameters
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def config_to_experiment_params(config: dict) -> ExperimentParams:
    """Convert YAML config to ExperimentParams object."""
    cloud_cfg = config['cloud']
    timing_cfg = config['timing']
    imaging_cfg = config['imaging']
    initial_pos_cfg = config['initial_position']
    
    # Explicitly convert to correct types
    return ExperimentParams(
        n_atoms=float(cloud_cfg['n_atoms']),
        n_atoms_uncertainty=float(cloud_cfg['n_atoms_uncertainty']),
        temperature=float(cloud_cfg['temperature_nK']) * 1e-9,
        temperature_min=float(cloud_cfg['temperature_min_nK']) * 1e-9,
        temperature_max=float(cloud_cfg['temperature_max_nK']) * 1e-9,
        temperature_fluctuation=float(cloud_cfg['temperature_fluctuation_nK']) * 1e-9,
        release_height=float(initial_pos_cfg['release_height_um']) * 1e-6,
        population_mF2=float(cloud_cfg['population']['mF_2_fraction']),
        population_mF1=float(cloud_cfg['population']['mF_1_fraction']),
        population_uncertainty=float(cloud_cfg['population']['uncertainty']),
        pixel_size=float(imaging_cfg['pixel_size_um']) * 1e-6,
        magnification=float(imaging_cfg['magnification']),
        observation_times=tuple(float(t) * 1e-3 for t in timing_cfg['observation_times_ms'])
    )


def setup_output_directories(output_base: Path) -> dict:
    """Create output directories and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dirs = {
        'base': output_base,
        'figures': output_base / 'figures' / f'free_fall_{timestamp}',
        'data': output_base / 'simulations' / f'free_fall_{timestamp}',
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def plot_trajectory(
    result: FreeFallSimulation,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot cloud trajectory (position vs time).
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    times_ms = result.times_ms
    pos_um = result.positions_um
    vel_mm_s = result.velocities * 1e3

    # Position plot
    ax1 = axes[0]
    ax1.plot(times_ms, pos_um[:, 0], 'b-', label='x (radial)', alpha=0.7)
    ax1.plot(times_ms, pos_um[:, 1], 'g-', label='y (radial)', alpha=0.7)
    ax1.plot(times_ms, pos_um[:, 2], 'r-', label='z (vertical)', linewidth=2)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Position (µm)')
    ax1.legend(loc='upper right')
    ax1.set_title('Cloud Center Position vs Time')
    ax1.grid(True, alpha=0.3)

    # Add collision time markers
    collision_times_ms = [5.6, 5.9, 6.4]  # For a=5, 10, 20
    for t_col in collision_times_ms:
        if t_col <= times_ms.max():
            ax1.axvline(t_col, color='orange', linestyle=':', alpha=0.5)

    # Velocity plot
    ax2 = axes[1]
    ax2.plot(times_ms, vel_mm_s[:, 2], 'r-',
             label='vz (vertical)', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Velocity (mm/s)')
    ax2.legend(loc='upper right')
    ax2.set_title('Cloud Center Velocity vs Time')
    ax2.grid(True, alpha=0.3)

    # Add theoretical free fall
    t_theory = result.times
    v_theory = -G * t_theory * 1e3  # mm/s
    ax2.plot(times_ms, v_theory, 'k--', alpha=0.5, label='v = -gt (theory)')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_expansion(
    result: FreeFallSimulation,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot cloud size expansion vs time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times_ms = result.times_ms
    sigmas_um = result.sigmas_um

    ax.plot(times_ms, sigmas_um[:, 0], 'b-', label='σx (radial)', linewidth=2)
    ax.plot(times_ms, sigmas_um[:, 1], 'g-', label='σy (radial)', linewidth=2)
    ax.plot(times_ms, sigmas_um[:, 2], 'r-',
            label='σz (longitudinal)', linewidth=2)

    # Average radial
    sigma_radial = (sigmas_um[:, 0] + sigmas_um[:, 1]) / 2
    ax.plot(times_ms, sigma_radial, 'k--', label='σ_radial (avg)', alpha=0.7)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cloud Size σ (µm)')
    ax.set_title('Cloud Expansion During Free Fall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with temperature
    temp_nK = result.params.get('temperature_nK', 295)
    textstr = f'T = {temp_nK:.0f} nK'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved expansion plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_profiles_at_times(
    cloud,
    times: np.ndarray,
    extent_um: float = 400,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot density profiles at specified observation times.
    """
    n_times = len(times)
    fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))

    if n_times == 1:
        axes = axes.reshape(2, 1)

    # Create a fresh cloud for each time
    from ultracold.simulation.cloud import MixedStateCloud

    extent_m = extent_um * 1e-6

    for i, t in enumerate(times):
        # Create fresh cloud and evolve to time t
        test_cloud = cloud.copy() if hasattr(cloud, 'copy') else create_experiment_cloud()
        if not test_cloud.is_released:
            test_cloud.release()
        test_cloud.evolve(t)

        t_ms = t * 1e3

        # Extract profiles
        profile_x = profile_from_simulation(
            test_cloud, 'x', n_points=200, extent=extent_m)
        profile_z = profile_from_simulation(
            test_cloud, 'z', n_points=200, extent=extent_m)

        # Horizontal (x) profile
        ax_h = axes[0, i]
        ax_h.plot(profile_x.position * 1e6, profile_x.values /
                  profile_x.values.max(), 'b-', linewidth=2)
        ax_h.set_xlabel('x (µm)')
        ax_h.set_ylabel('Normalized Density')
        ax_h.set_title(f't = {t_ms:.1f} ms\n(Radial)')
        ax_h.grid(True, alpha=0.3)
        ax_h.set_xlim(-extent_um, extent_um)

        # Vertical (z) profile
        ax_v = axes[1, i]
        ax_v.plot(profile_z.position * 1e6, profile_z.values /
                  profile_z.values.max(), 'r-', linewidth=2)
        ax_v.set_xlabel('z (µm)')
        ax_v.set_ylabel('Normalized Density')
        ax_v.set_title(f'(Vertical/Longitudinal)')
        ax_v.grid(True, alpha=0.3)
        ax_v.set_xlim(-extent_um, extent_um)

        # Add cloud size info
        if hasattr(test_cloud, 'clouds'):
            sigma = test_cloud.clouds[0].sigma
        else:
            sigma = test_cloud.sigma

        sigma_x_um = sigma[0] * 1e6
        sigma_z_um = sigma[2] * 1e6

        ax_h.text(0.95, 0.95, f'σx={sigma_x_um:.1f}µm', transform=ax_h.transAxes,
                  ha='right', va='top', fontsize=9)
        ax_v.text(0.95, 0.95, f'σz={sigma_z_um:.1f}µm', transform=ax_v.transAxes,
                  ha='right', va='top', fontsize=9)

    plt.suptitle('Density Profiles at Observation Times', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved profiles plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_simulated_images(
    cloud,
    times: np.ndarray,
    pixel_size: float = 3e-6,
    n_pixels: int = 256,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate and plot simulated absorption images at specified times.
    """
    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(5*n_times, 5))

    if n_times == 1:
        axes = [axes]

    # Coordinate grid
    extent_m = n_pixels * pixel_size / 2
    x = np.linspace(-extent_m, extent_m, n_pixels)
    y = np.linspace(-extent_m, extent_m, n_pixels)
    X, Y = np.meshgrid(x, y)

    extent_um = [-extent_m*1e6, extent_m*1e6, -extent_m*1e6, extent_m*1e6]

    for i, t in enumerate(times):
        # Create fresh cloud and evolve
        test_cloud = create_experiment_cloud()
        test_cloud.release()
        test_cloud.evolve(t)

        # Generate image
        image = test_cloud.generate_image(X, Y, noise_level=0.02)

        t_ms = t * 1e3

        ax = axes[i]
        im = ax.imshow(image, extent=extent_um, origin='lower', cmap='viridis')
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_title(f't = {t_ms:.1f} ms')
        plt.colorbar(im, ax=ax, label='Column Density', shrink=0.8)

        # Mark center
        if hasattr(test_cloud, 'clouds'):
            pos = test_cloud.clouds[0].position
        else:
            pos = test_cloud.position
        ax.plot(pos[0]*1e6, pos[1]*1e6, 'r+', markersize=15, markeredgewidth=2)

    plt.suptitle('Simulated Absorption Images', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved images plot to {save_path}")

    if show:
        plt.show()

    return fig


def print_cloud_info(cloud, title: str = "Cloud Information"):
    """Print detailed cloud information."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

    if hasattr(cloud, 'clouds'):  # MixedStateCloud
        print(f"Type: Mixed State Cloud")
        print(f"Total atoms: {cloud.n_atoms_total:.2e}")
        print(f"Temperature: {cloud.temperature*1e9:.1f} nK")
        print(f"States:")
        for c in cloud.clouds:
            print(f"  |F={c.state.F}, mF={c.state.mF}⟩: {
                  c.n_atoms:.2e} atoms ({c.n_atoms/cloud.n_atoms_total*100:.0f}%)")
            print(f"    Trap freq (radial): {
                  c.state.omega_radial/(2*np.pi):.1f} Hz")
            print(f"    Trap freq (longitudinal): {
                  c.state.omega_longitudinal/(2*np.pi):.1f} Hz")
            sigma_um = c._initial_sigma * 1e6
            print(f"    Initial size: σ = ({sigma_um[0]:.1f}, {
                  sigma_um[1]:.1f}, {sigma_um[2]:.1f}) µm")
    else:  # ThermalCloud
        print(f"Type: Single State Thermal Cloud")
        print(f"Atoms: {cloud.n_atoms:.2e}")
        print(f"Temperature: {cloud.temperature*1e9:.1f} nK")
        print(f"State: |F={cloud.state.F}, mF={cloud.state.mF}⟩")
        sigma_um = cloud._initial_sigma * 1e6
        print(f"Initial size: σ = ({sigma_um[0]:.1f}, {
              sigma_um[1]:.1f}, {sigma_um[2]:.1f}) µm")

    # Thermal velocity
    v_th = np.sqrt(KB * cloud.temperature / M_RB87)
    print(f"\nThermal velocity: {v_th*1e3:.2f} mm/s")

    # Initial position
    if hasattr(cloud, 'clouds'):
        pos = cloud.clouds[0].initial_position
    else:
        pos = cloud.initial_position
    print(f"Release position: ({
          pos[0]*1e6:.1f}, {pos[1]*1e6:.1f}, {pos[2]*1e6:.1f}) µm")

    print('='*60)


def print_results_summary(result: FreeFallSimulation, observation_times: np.ndarray):
    """Print summary of simulation results."""
    print(f"\n{'='*60}")
    print(" SIMULATION RESULTS SUMMARY")
    print('='*60)

    print(f"\nSimulation time range: 0 to {result.times[-1]*1e3:.1f} ms")
    print(f"Number of time points: {len(result.times)}")

    print(f"\nAt observation times:")
    print("-" * 60)
    print(f"{'Time (ms)':<12} {'z (µm)':<12} {
          'vz (mm/s)':<12} {'σ_rad (µm)':<12} {'σ_z (µm)':<12}")
    print("-" * 60)

    for t_obs in observation_times:
        # Find closest time index
        idx = np.argmin(np.abs(result.times - t_obs))

        t_ms = result.times[idx] * 1e3
        z_um = result.positions[idx, 2] * 1e6
        vz_mm_s = result.velocities[idx, 2] * 1e3
        sigma_rad = (result.sigmas[idx, 0] + result.sigmas[idx, 1]) / 2 * 1e6
        sigma_z = result.sigmas[idx, 2] * 1e6

        print(f"{t_ms:<12.1f} {z_um:<12.1f} {vz_mm_s:<12.2f} {
              sigma_rad:<12.1f} {sigma_z:<12.1f}")

    print("-" * 60)

    # Final state
    print(f"\nFinal state (t = {result.times[-1]*1e3:.1f} ms):")
    print(f"  Position: ({result.positions[-1, 0]*1e6:.1f}, {
          result.positions[-1, 1]*1e6:.1f}, {result.positions[-1, 2]*1e6:.1f}) µm")
    print(f"  Velocity: ({result.velocities[-1, 0]*1e3:.2f}, {
          result.velocities[-1, 1]*1e3:.2f}, {result.velocities[-1, 2]*1e3:.2f}) mm/s")
    print(f"  Size σ: ({result.sigmas[-1, 0]*1e6:.1f}, {
          result.sigmas[-1, 1]*1e6:.1f}, {result.sigmas[-1, 2]*1e6:.1f}) µm")

    print('='*60)


def main():
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

    # Print header
    print("\n" + "="*60)
    print(" ULTRACOLD ATOMS FREE FALL SIMULATION")
    print(" Rb-87 in Ioffe Trap")
    print("="*60)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    try:
        config = load_config(args.config)
        params = config_to_experiment_params(config)
        print("Configuration loaded successfully.")
    except FileNotFoundError:
        print("Config file not found. Using default parameters.")
        params = ExperimentParams()
        config = None

    # Setup output directories
    if not args.no_save:
        output_base = Path(
            args.output_dir) if args.output_dir else PROJECT_ROOT / 'output'
        dirs = setup_output_directories(output_base)
        print(f"Output directory: {dirs['data']}")

    # Create cloud
    print("\nCreating atomic cloud...")
    if args.single_state:
        state = STATE_F2_MF2 if args.state == 'mF2' else STATE_F2_MF1
        cloud = ThermalCloud(
            n_atoms=params.n_atoms,
            temperature=params.temperature,
            state=state,
            position=np.array([0, 0, params.release_height])
        )
    else:
        cloud = create_experiment_cloud(params=params, mixed_state=True)

    print_cloud_info(cloud)

    # Setup time array
    t_max = args.t_max * 1e-3  # Convert to seconds
    times = np.linspace(0, t_max, args.n_points)

    # Observation times from config or defaults
    if config:
        observation_times = np.array(
            config['timing']['observation_times_ms']) * 1e-3
    else:
        observation_times = np.array(params.observation_times)

    print(f"\nObservation times: {observation_times*1e3} ms")

    # Run simulation
    print("\nRunning free fall simulation...")
    result = simulate_free_fall(cloud, times, release=True)
    print("Simulation complete.")

    # Print results summary
    print_results_summary(result, observation_times)

    # Save results
    if not args.no_save:
        # Save to CSV
        csv_path = dirs['data'] / 'free_fall_results.csv'
        result.save(str(csv_path))

        # Save parameters
        params_path = dirs['data'] / 'simulation_params.yaml'
        with open(params_path, 'w') as f:
            yaml.dump({
                'simulation': {
                    't_max_ms': args.t_max,
                    'n_points': args.n_points,
                    'single_state': args.single_state,
                    'state': args.state if args.single_state else 'mixed',
                },
                'cloud_params': result.params,
                'observation_times_ms': (observation_times * 1e3).tolist(),
            }, f, default_flow_style=False)
        print(f"Saved parameters to {params_path}")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")

        # Trajectory plot
        traj_path = dirs['figures'] / \
            'trajectory.png' if not args.no_save else None
        plot_trajectory(result, save_path=traj_path, show=True)

        # Expansion plot
        exp_path = dirs['figures'] / \
            'expansion.png' if not args.no_save else None
        plot_expansion(result, save_path=exp_path, show=True)

        # Profiles at observation times
        prof_path = dirs['figures'] / \
            'profiles.png' if not args.no_save else None
        plot_profiles_at_times(cloud, observation_times,
                               save_path=prof_path, show=True)

        # Simulated images
        img_path = dirs['figures'] / \
            'simulated_images.png' if not args.no_save else None
        plot_simulated_images(cloud, observation_times,
                              save_path=img_path, show=True)

    print("\n" + "="*60)
    print(" SIMULATION COMPLETE")
    print("="*60)

    return result, cloud


if __name__ == '__main__':
    result, cloud = main()
