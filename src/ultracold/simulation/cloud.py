# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:20:33 2026

@author: clher
"""
"""
Ultracold atomic cloud simulation.

Classes for thermal clouds of Rb-87 atoms with support for:
- Multiple quantum states (mF=1, mF=2)
- Anisotropic trap geometries
- Time-of-flight expansion
- Free fall dynamics
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import copy

from ..utils.constants import (
    M_RB87, KB, G, HBAR,
    STATE_F2_MF2, STATE_F2_MF1,
    thermal_velocity, thermal_size,
    Rb87State, ExperimentParams, DEFAULT_PARAMS
)


# =============================================================================
# CLOUD STATE CONTAINER
# =============================================================================

@dataclass
class CloudState:
    """
    Snapshot of cloud state at a given time.
    
    Stores position, size, and velocity information.
    """
    time: float                           # Time since release [s]
    position: np.ndarray                  # Center position (x, y, z) [m]
    velocity: np.ndarray                  # Center velocity (vx, vy, vz) [m/s]
    sigma: np.ndarray                     # Cloud sizes (σx, σy, σz) [m]
    n_atoms: float                        # Number of atoms
    temperature: float                    # Temperature [K]
    
    def __repr__(self):
        t_ms = self.time * 1e3
        pos_um = self.position * 1e6
        sig_um = self.sigma * 1e6
        return (f"CloudState(t={t_ms:.2f}ms, "
                f"pos=[{pos_um[0]:.1f}, {pos_um[1]:.1f}, {pos_um[2]:.1f}]µm, "
                f"σ=[{sig_um[0]:.1f}, {sig_um[1]:.1f}, {sig_um[2]:.1f}]µm)")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            'time_s': self.time,
            'time_ms': self.time * 1e3,
            'position_m': self.position.tolist(),
            'position_um': (self.position * 1e6).tolist(),
            'velocity_m_s': self.velocity.tolist(),
            'sigma_m': self.sigma.tolist(),
            'sigma_um': (self.sigma * 1e6).tolist(),
            'n_atoms': self.n_atoms,
            'temperature_K': self.temperature,
            'temperature_nK': self.temperature * 1e9,
        }


# =============================================================================
# THERMAL CLOUD CLASS
# =============================================================================

class ThermalCloud:
    """
    Thermal cloud of ultracold Rb-87 atoms.
    
    Simulates a Gaussian thermal cloud in a harmonic trap with support
    for free expansion and free fall after release.
    
    Parameters
    ----------
    n_atoms : float
        Number of atoms
    temperature : float
        Cloud temperature [K]
    state : Rb87State
        Quantum state with trap frequencies
    position : array-like, optional
        Initial center position (x, y, z) [m]
    velocity : array-like, optional
        Initial center velocity (vx, vy, vz) [m/s]
    
    Examples
    --------
    >>> cloud = ThermalCloud(
    ...     n_atoms=1.2e5,
    ...     temperature=295e-9,
    ...     state=STATE_F2_MF2,
    ...     position=[0, 0, 140e-6]
    ... )
    >>> cloud.release()
    >>> state_at_7ms = cloud.evolve(7e-3)
    """
    
    def __init__(
        self,
        n_atoms: float,
        temperature: float,
        state: Rb87State,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None
    ):
        # Store parameters
        self.n_atoms = n_atoms
        self.temperature = temperature
        self.state = state
        self.mass = M_RB87
        
        # Initial position (default: origin)
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self.initial_position = np.array(position, dtype=float)
        
        # Initial velocity (default: at rest)
        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0])
        self.initial_velocity = np.array(velocity, dtype=float)
        
        # Current state
        self._position = self.initial_position.copy()
        self._velocity = self.initial_velocity.copy()
        self._time = 0.0
        self._released = False
        
        # Calculate initial cloud sizes in trap
        self._initial_sigma = self._calculate_trapped_size()
        self._sigma = self._initial_sigma.copy()
        
        # Thermal velocity for expansion
        self._v_thermal = thermal_velocity(self.temperature, self.mass)
        
        # History for tracking evolution
        self.history: List[CloudState] = []
    
    def _calculate_trapped_size(self) -> np.ndarray:
        """Calculate initial cloud size in the trap."""
        omega_x, omega_y, omega_z = self.state.trap_frequencies
        
        sigma_x = thermal_size(self.temperature, omega_x, self.mass)
        sigma_y = thermal_size(self.temperature, omega_y, self.mass)
        sigma_z = thermal_size(self.temperature, omega_z, self.mass)
        
        return np.array([sigma_x, sigma_y, sigma_z])
    
    @property
    def position(self) -> np.ndarray:
        """Current center position [m]."""
        return self._position
    
    @property
    def velocity(self) -> np.ndarray:
        """Current center velocity [m/s]."""
        return self._velocity
    
    @property
    def sigma(self) -> np.ndarray:
        """Current cloud sizes (σx, σy, σz) [m]."""
        return self._sigma
    
    @property
    def time(self) -> float:
        """Current time since release [s]."""
        return self._time
    
    @property
    def is_released(self) -> bool:
        """Whether the cloud has been released from the trap."""
        return self._released
    
    @property
    def fwhm(self) -> np.ndarray:
        """Full width at half maximum [m]."""
        return 2.355 * self._sigma  # FWHM = 2√(2ln2) × σ ≈ 2.355σ
    
    def release(self):
        """Release the cloud from the trap (start free fall)."""
        self._released = True
        self._time = 0.0
        self._record_state()
    
    def evolve(self, t: float) -> CloudState:
        """
        Evolve cloud to time t after release.
        
        Parameters
        ----------
        t : float
            Time after release [s]
        
        Returns
        -------
        CloudState
            Cloud state at time t
        """
        if not self._released:
            raise RuntimeError("Cloud must be released before evolving. Call release() first.")
        
        # Update time
        self._time = t
        
        # Position: free fall in z (assuming z is vertical, gravity in -z)
        self._position = self.initial_position + self.initial_velocity * t
        self._position[2] -= 0.5 * G * t**2  # z = z0 + v0*t - 0.5*g*t²
        
        # Velocity
        self._velocity = self.initial_velocity.copy()
        self._velocity[2] -= G * t  # vz = v0z - g*t
        
        # Cloud expansion: σ(t) = √(σ₀² + v_th² × t²)
        self._sigma = np.sqrt(self._initial_sigma**2 + (self._v_thermal * t)**2)
        
        # Record state
        state = self._get_current_state()
        self._record_state()
        
        return state
    
    def evolve_times(self, times: np.ndarray) -> List[CloudState]:
        """
        Evolve cloud through multiple times.
        
        Parameters
        ----------
        times : array-like
            Array of times after release [s]
        
        Returns
        -------
        list of CloudState
            Cloud states at each time
        """
        if not self._released:
            self.release()
        
        states = []
        for t in times:
            states.append(self.evolve(t))
        
        return states
    
    def _get_current_state(self) -> CloudState:
        """Get current state as CloudState object."""
        return CloudState(
            time=self._time,
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            sigma=self._sigma.copy(),
            n_atoms=self.n_atoms,
            temperature=self.temperature
        )
    
    def _record_state(self):
        """Record current state to history."""
        self.history.append(self._get_current_state())
    
    def density_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculate 3D density distribution.
        
        Parameters
        ----------
        x, y, z : ndarray
            Coordinate grids [m]
        
        Returns
        -------
        ndarray
            Density [atoms/m³]
        """
        sx, sy, sz = self._sigma
        x0, y0, z0 = self._position
        
        norm = self.n_atoms / ((2 * np.pi)**1.5 * sx * sy * sz)
        
        return norm * np.exp(
            -(x - x0)**2 / (2 * sx**2)
            -(y - y0)**2 / (2 * sy**2)
            -(z - z0)**2 / (2 * sz**2)
        )
    
    def column_density(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        integration_axis: str = 'z'
    ) -> np.ndarray:
        """
        Calculate column density (integrated along one axis).
        
        This is what you typically measure in absorption imaging.
        
        Parameters
        ----------
        x, y : ndarray
            Coordinate grids [m] (can be meshgrid or 1D arrays)
        integration_axis : str
            Axis to integrate along ('x', 'y', or 'z')
        
        Returns
        -------
        ndarray
            Column density [atoms/m²]
        """
        sx, sy, sz = self._sigma
        x0, y0, z0 = self._position
        
        if integration_axis == 'z':
            # Integrate along z → 2D Gaussian in x-y
            norm = self.n_atoms / (2 * np.pi * sx * sy)
            return norm * np.exp(
                -(x - x0)**2 / (2 * sx**2)
                -(y - y0)**2 / (2 * sy**2)
            )
        elif integration_axis == 'y':
            # Integrate along y → 2D Gaussian in x-z
            norm = self.n_atoms / (2 * np.pi * sx * sz)
            return norm * np.exp(
                -(x - x0)**2 / (2 * sx**2)
                -(y - z0)**2 / (2 * sz**2)
            )
        elif integration_axis == 'x':
            # Integrate along x → 2D Gaussian in y-z
            norm = self.n_atoms / (2 * np.pi * sy * sz)
            return norm * np.exp(
                -(x - y0)**2 / (2 * sy**2)
                -(y - z0)**2 / (2 * sz**2)
            )
        else:
            raise ValueError(f"integration_axis must be 'x', 'y', or 'z', got '{integration_axis}'")
    
    def generate_image(
        self,
        x: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.0,
        photon_shot_noise: bool = False,
        integration_axis: str = 'z'
    ) -> np.ndarray:
        """
        Generate simulated absorption image.
        
        Parameters
        ----------
        x, y : ndarray
            Coordinate grids [m]
        noise_level : float
            Gaussian noise level (as fraction of peak signal)
        photon_shot_noise : bool
            Add Poisson shot noise
        integration_axis : str
            Direction of imaging beam
        
        Returns
        -------
        ndarray
            Simulated image (column density + noise)
        """
        image = self.column_density(x, y, integration_axis)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * image.max(), image.shape)
            image = image + noise
        
        if photon_shot_noise:
            image = np.random.poisson(np.maximum(image, 0).astype(int)).astype(float)
        
        return image
    
    def copy(self) -> 'ThermalCloud':
        """Create a deep copy of the cloud."""
        return copy.deepcopy(self)
    
    def __repr__(self):
        status = "released" if self._released else "trapped"
        return (f"ThermalCloud(N={self.n_atoms:.2e}, T={self.temperature*1e9:.0f}nK, "
                f"state=|F={self.state.F}, mF={self.state.mF}⟩, {status})")


# =============================================================================
# MIXED STATE CLOUD CLASS
# =============================================================================

class MixedStateCloud:
    """
    Cloud with atoms in multiple quantum states.
    
    For your experiment: 50% in |F=2, mF=2⟩ and 50% in |F=2, mF=1⟩
    
    Parameters
    ----------
    n_atoms_total : float
        Total number of atoms
    temperature : float
        Cloud temperature [K]
    populations : dict, optional
        Dictionary mapping Rb87State to population fraction
    position : array-like, optional
        Initial center position
    
    Examples
    --------
    >>> cloud = MixedStateCloud(
    ...     n_atoms_total=1.2e5,
    ...     temperature=295e-9,
    ...     position=[0, 0, 140e-6]
    ... )
    """
    
    def __init__(
        self,
        n_atoms_total: float,
        temperature: float,
        populations: Optional[dict] = None,
        position: Optional[np.ndarray] = None
    ):
        self.n_atoms_total = n_atoms_total
        self.temperature = temperature
        
        # Default: 50-50 split between mF=2 and mF=1
        if populations is None:
            populations = {
                STATE_F2_MF2: 0.5,
                STATE_F2_MF1: 0.5
            }
        
        self.populations = populations
        
        # Create individual clouds for each state
        self.clouds: List[ThermalCloud] = []
        for state, fraction in populations.items():
            cloud = ThermalCloud(
                n_atoms=n_atoms_total * fraction,
                temperature=temperature,
                state=state,
                position=position
            )
            self.clouds.append(cloud)
        
        self._released = False
        self.history: List[dict] = []
    
    @property
    def is_released(self) -> bool:
        """Whether the cloud has been released from the trap."""
        return self._released
    
    @property
    def position(self) -> np.ndarray:
        """Average position (all clouds should have same center)."""
        return self.clouds[0].position
    
    @property
    def sigma(self) -> np.ndarray:
        """Average sigma across all states (weighted by atom number)."""
        total_atoms = sum(c.n_atoms for c in self.clouds)
        avg_sigma = sum(c.sigma * c.n_atoms for c in self.clouds) / total_atoms
        return avg_sigma
    
    @property
    def time(self) -> float:
        """Current time."""
        return self.clouds[0].time
    
    @property
    def initial_position(self) -> np.ndarray:
        """Initial position of the cloud."""
        return self.clouds[0].initial_position
    
    @property
    def initial_velocity(self) -> np.ndarray:
        """Initial velocity of the cloud."""
        return self.clouds[0].initial_velocity
    
    def release(self):
        """Release all clouds from the trap."""
        for cloud in self.clouds:
            cloud.release()
        self._released = True
    
    def evolve(self, t: float) -> List[CloudState]:
        """Evolve all clouds to time t."""
        if not self._released:
            self.release()
        
        states = []
        for cloud in self.clouds:
            states.append(cloud.evolve(t))
        
        return states
    
    def combined_column_density(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        integration_axis: str = 'z'
    ) -> np.ndarray:
        """
        Calculate combined column density from all states.
        
        Note: Different states may have different sizes due to different
        trap frequencies!
        """
        total_density = np.zeros_like(x, dtype=float)
        
        for cloud in self.clouds:
            total_density += cloud.column_density(x, y, integration_axis)
        
        return total_density
    
    def generate_image(
        self,
        x: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.0,
        integration_axis: str = 'z'
    ) -> np.ndarray:
        """Generate combined image from all states."""
        image = self.combined_column_density(x, y, integration_axis)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * image.max(), image.shape)
            image = image + noise
        
        return image
    
    def copy(self) -> 'MixedStateCloud':
        """Create a deep copy of the cloud."""
        return copy.deepcopy(self)
    
    def __repr__(self):
        states_str = ", ".join(
            f"mF={c.state.mF}:{c.n_atoms/self.n_atoms_total*100:.0f}%"
            for c in self.clouds
        )
        status = "released" if self._released else "trapped"
        return f"MixedStateCloud(N={self.n_atoms_total:.2e}, T={self.temperature*1e9:.0f}nK, [{states_str}], {status})"


# =============================================================================
# BEC CLOUD CLASS (for future use)
# =============================================================================

class BECCloud(ThermalCloud):
    """
    Bose-Einstein Condensate with Thomas-Fermi profile.
    
    For pure BEC at T=0.
    """
    
    def __init__(
        self,
        n_atoms: float,
        chemical_potential: float,
        state: Rb87State,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None
    ):
        # Initialize with T=0
        super().__init__(
            n_atoms=n_atoms,
            temperature=0,
            state=state,
            position=position,
            velocity=velocity
        )
        self.mu = chemical_potential
        
        # Override initial sigma with Thomas-Fermi radii
        self._initial_sigma = self._calculate_tf_radii()
        self._sigma = self._initial_sigma.copy()
    
    def _calculate_tf_radii(self) -> np.ndarray:
        """Calculate Thomas-Fermi radii."""
        omega_x, omega_y, omega_z = self.state.trap_frequencies
        
        r_x = np.sqrt(2 * self.mu / (self.mass * omega_x**2))
        r_y = np.sqrt(2 * self.mu / (self.mass * omega_y**2))
        r_z = np.sqrt(2 * self.mu / (self.mass * omega_z**2))
        
        return np.array([r_x, r_y, r_z])
    
    @property
    def thomas_fermi_radii(self) -> np.ndarray:
        """Thomas-Fermi radii."""
        return self._calculate_tf_radii()
    
    def column_density(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        integration_axis: str = 'z'
    ) -> np.ndarray:
        """Thomas-Fermi column density profile (integrated parabola)."""
        r_x, r_y, r_z = self._sigma  # Use current sigma (may have expanded)
        x0, y0, z0 = self._position
        
        if integration_axis == 'z':
            # Parabolic profile integrated along z
            arg = 1 - ((x - x0)/r_x)**2 - ((y - y0)/r_y)**2
            arg = np.maximum(arg, 0)
            
            # Normalization for TF profile
            norm = 15 * self.n_atoms / (8 * np.pi * r_x * r_y)
            
            return norm * arg**1.5
        else:
            # For other axes, fall back to Gaussian approximation
            return super().column_density(x, y, integration_axis)