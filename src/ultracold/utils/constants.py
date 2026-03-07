# -*- coding: utf-8 -*-

"""
Physical constants and experimental parameters for Rb-87 ultracold atoms.

All values in SI units unless otherwise noted.
Based on experimental parameters from Weber beam experiment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

HBAR = 1.054571817e-34      # Reduced Planck constant [J·s]
KB = 1.380649e-23           # Boltzmann constant [J/K]
C = 299792458.0             # Speed of light [m/s]
G = 9.81                    # Gravitational acceleration [m/s²]
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]

# =============================================================================
# RUBIDIUM-87 PROPERTIES
# =============================================================================

# Mass
M_RB87 = 1.443160648e-25    # Mass of Rb-87 [kg]
M_RB87_AMU = 86.909180520   # Mass in atomic mass units

# D2 Line (5S_1/2 → 5P_3/2)
LAMBDA_D2 = 780.241209686e-9    # D2 transition wavelength [m]
FREQ_D2 = C / LAMBDA_D2         # D2 transition frequency [Hz]
OMEGA_D2 = 2 * np.pi * FREQ_D2  # D2 angular frequency [rad/s]
# Natural linewidth [rad/s] (Γ = 2π × 6.07 MHz)
GAMMA_D2 = 2 * np.pi * 6.0666e6
TAU_D2 = 1 / GAMMA_D2           # Excited state lifetime [s]

# Saturation intensity for D2 line (cycling transition)
I_SAT_D2 = 1.6693e-3 * 1e4      # Saturation intensity [W/m²] ≈ 1.67 mW/cm²


# =============================================================================
# EXPERIMENTAL PARAMETERS (from your experiment)
# =============================================================================

@dataclass(frozen=True)
class Rb87State:
    """
    Quantum state parameters for Rb-87 in a magnetic trap.

    Attributes
    ----------
    F : int
        Total angular momentum quantum number
    mF : int
        Magnetic quantum number
    omega_longitudinal : float
        Longitudinal trap frequency [rad/s]
    omega_radial : float
        Radial trap frequency [rad/s]
    """
    F: int
    mF: int
    omega_longitudinal: float  # rad/s
    omega_radial: float        # rad/s

    @property
    def trap_frequencies(self) -> Tuple[float, float, float]:
        """Return (ω_x, ω_y, ω_z) assuming radial symmetry in x-y plane."""
        return (self.omega_radial, self.omega_radial, self.omega_longitudinal)

    @property
    def frequencies_hz(self) -> Tuple[float, float, float]:
        """Return trap frequencies in Hz."""
        return tuple(f / (2 * np.pi) for f in self.trap_frequencies)


# Pre-defined states from your experiment
STATE_F2_MF2 = Rb87State(
    F=2,
    mF=2,
    omega_longitudinal=2 * np.pi * 45.0,    # 45 Hz
    omega_radial=2 * np.pi * 247.0           # 247 Hz
)

STATE_F2_MF1 = Rb87State(
    F=2,
    mF=1,
    omega_longitudinal=2 * np.pi * 31.8,    # 31.8 Hz
    omega_radial=2 * np.pi * 174.5           # 174.5 Hz
)


@dataclass
class ExperimentParams:
    """
    Container for experimental parameters.

    Default values from your Weber beam experiment with Rb-87.
    """
    # Atom numbers
    n_atoms: float = 1.2e5
    n_atoms_uncertainty: float = 0.2e5

    # Temperature
    temperature: float = 295e-9           # 295 nK (middle of 280-310 range)
    temperature_min: float = 280e-9       # Day-to-day minimum
    temperature_max: float = 310e-9       # Day-to-day maximum
    temperature_fluctuation: float = 5e-9  # Shot-to-shot fluctuation

    # Initial position (release height)
    release_height: float = 140e-6        # 140 µm

    # Population in each state
    population_mF2: float = 0.5           # 50% in mF=2
    population_mF1: float = 0.5           # 50% in mF=1
    population_uncertainty: float = 0.1   # ±10%

    # Imaging parameters (you may need to adjust these)
    pixel_size: float = 6.45e-6           # Camera pixel size [m]
    magnification: float = 1.0            # Imaging magnification

    # Observation times from experiment
    observation_times: Tuple[float, ...] = field(
        default_factory=lambda: (7e-3, 13e-3, 18e-3)  # 7, 13, 18 ms
    )

    # Weber beam collision times
    collision_times: dict = field(default_factory=lambda: {
        5: 5.6e-3,    # a=5:  5.6 ms
        10: 5.9e-3,   # a=10: 5.9 ms
        20: 6.4e-3    # a=20: 6.4 ms
    })

    @property
    def effective_pixel_size(self) -> float:
        """Pixel size in object plane [m]."""
        return self.pixel_size / self.magnification

    def get_random_temperature(self, shot_to_shot: bool = True) -> float:
        """
        Get temperature with realistic fluctuations.

        Parameters
        ----------
        shot_to_shot : bool
            If True, add shot-to-shot fluctuation to base temperature
        """
        base_temp = np.random.uniform(
            self.temperature_min, self.temperature_max)
        if shot_to_shot:
            base_temp += np.random.normal(0, self.temperature_fluctuation)
        return base_temp


@dataclass
class WeberBeamParams:
    """
    Weber beam parameters.
    """
    wavelength: float = 780.368e-9        # Beam wavelength [m]
    red_detuning: float = 0.127e-9        # Red detuning from D2 [m]

    # Weber parameters studied
    weber_params: Tuple[int, ...] = (5, 10, 20)

    # Lobe separations (dark region widths)
    lobe_separations: dict = field(default_factory=lambda: {
        5: 36e-6,     # 36 µm for a=5
        10: 66e-6,    # 66 µm for a=10
        20: 121e-6    # 121 µm for a=20
    })

    # Rabi frequencies (in units of Γ)
    rabi_frequencies: dict = field(default_factory=lambda: {
        5: 19.1,      # g/Γ for a=5
        10: 18.5,     # g/Γ for a=10
        20: 18.5      # g/Γ for a=20
    })

    # Transverse wave number
    k_perp: float = 0.375e6               # 0.375 µm⁻¹ = 0.375×10⁶ m⁻¹

    # Geometry
    angle_to_trap_axis: float = np.radians(80)  # 80° angle

    @property
    def detuning_hz(self) -> float:
        """Detuning in Hz."""
        return C * self.red_detuning / self.wavelength**2

    @property
    def detuning_gamma(self) -> float:
        """Detuning in units of Γ."""
        return 2 * np.pi * self.detuning_hz / GAMMA_D2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def thermal_velocity(temperature: float, mass: float = M_RB87) -> float:
    """
    Calculate thermal velocity (1D RMS velocity).

    v_th = sqrt(k_B * T / m)

    Parameters
    ----------
    temperature : float
        Temperature [K]
    mass : float
        Atomic mass [kg], default is Rb-87

    Returns
    -------
    float
        Thermal velocity [m/s]
    """
    return np.sqrt(KB * temperature / mass)


def thermal_size(temperature: float, omega: float, mass: float = M_RB87) -> float:
    """
    Calculate thermal cloud size (1D Gaussian sigma) in harmonic trap.

    σ = sqrt(k_B * T / (m * ω²))

    Parameters
    ----------
    temperature : float
        Temperature [K]
    omega : float
        Trap angular frequency [rad/s]
    mass : float
        Atomic mass [kg]

    Returns
    -------
    float
        Cloud size (Gaussian sigma) [m]
    """
    return np.sqrt(KB * temperature / (mass * omega**2))


def cloud_size_after_tof(sigma_0: float, v_thermal: float, tof: float) -> float:
    """
    Calculate cloud size after time-of-flight expansion.

    σ(t) = sqrt(σ₀² + v_th² × t²)

    Parameters
    ----------
    sigma_0 : float
        Initial cloud size [m]
    v_thermal : float
        Thermal velocity [m/s]
    tof : float
        Time of flight [s]

    Returns
    -------
    float
        Expanded cloud size [m]
    """
    return np.sqrt(sigma_0**2 + (v_thermal * tof)**2)


def free_fall_position(z0: float, t: float, v0: float = 0) -> float:
    """
    Calculate position during free fall.

    z(t) = z₀ + v₀t - ½gt²

    Parameters
    ----------
    z0 : float
        Initial position [m]
    t : float
        Time [s]
    v0 : float
        Initial velocity [m/s], default 0

    Returns
    -------
    float
        Position at time t [m]
    """
    return z0 + v0 * t - 0.5 * G * t**2


def free_fall_velocity(t: float, v0: float = 0) -> float:
    """
    Calculate velocity during free fall.

    v(t) = v₀ - gt

    Parameters
    ----------
    t : float
        Time [s]
    v0 : float
        Initial velocity [m/s]

    Returns
    -------
    float
        Velocity at time t [m/s]
    """
    return v0 - G * t


# Quick access to default parameters
DEFAULT_PARAMS = ExperimentParams()
WEBER_PARAMS = WeberBeamParams()
