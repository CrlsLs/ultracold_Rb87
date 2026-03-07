# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:50:56 2026

@author: clher
"""

"""
Free fall simulation for ultracold atomic clouds.

Provides high-level functions for simulating and analyzing
the free fall evolution of thermal clouds.
"""


import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd

from .cloud import ThermalCloud, MixedStateCloud, CloudState
from ..utils.constants import (
    G, KB, M_RB87,
    STATE_F2_MF2, STATE_F2_MF1,
    ExperimentParams, DEFAULT_PARAMS
)



@dataclass
class FreeFallSimulation:
    """
    Container for free fall simulation results.

    Attributes
    ----------
    times : ndarray
        Time points [s]
    positions : ndarray
        Center positions at each time, shape (n_times, 3) [m]
    velocities : ndarray
        Center velocities at each time, shape (n_times, 3) [m/s]
    sigmas : ndarray
        Cloud sizes at each time, shape (n_times, 3) [m]
    states : list of CloudState
        Full state information at each time
    params : dict
        Simulation parameters
    """
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    sigmas: np.ndarray
    states: List[CloudState]
    params: dict

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def times_ms(self) -> np.ndarray:
        """Times in milliseconds."""
        return self.times * 1e3

    @property
    def positions_um(self) -> np.ndarray:
        """Positions in micrometers."""
        return self.positions * 1e6

    @property
    def sigmas_um(self) -> np.ndarray:
        """Cloud sizes in micrometers."""
        return self.sigmas * 1e6

    def get_position_z(self) -> np.ndarray:
        """Get z-position (vertical) vs time."""
        return self.positions[:, 2]

    def get_sigma_radial(self) -> np.ndarray:
        """Get average radial size (x-y average) vs time."""
        return (self.sigmas[:, 0] + self.sigmas[:, 1]) / 2

    def get_sigma_longitudinal(self) -> np.ndarray:
        """Get longitudinal size (z) vs time."""
        return self.sigmas[:, 2]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = {
            'time_ms': self.times_ms,
            'x_um': self.positions_um[:, 0],
            'y_um': self.positions_um[:, 1],
            'z_um': self.positions_um[:, 2],
            'vx_mm_s': self.velocities[:, 0] * 1e3,
            'vy_mm_s': self.velocities[:, 1] * 1e3,
            'vz_mm_s': self.velocities[:, 2] * 1e3,
            'sigma_x_um': self.sigmas_um[:, 0],
            'sigma_y_um': self.sigmas_um[:, 1],
            'sigma_z_um': self.sigmas_um[:, 2],
        }
        return pd.DataFrame(data)

    def save(self, filepath: str):
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Saved simulation results to {filepath}")


def simulate_free_fall(
    cloud: Union[ThermalCloud, MixedStateCloud],
    times: np.ndarray,
    release: bool = True
) -> FreeFallSimulation:
    """
    Simulate free fall of an atomic cloud.

    Parameters
    ----------
    cloud : ThermalCloud or MixedStateCloud
        The atomic cloud to simulate
    times : array-like
        Time points to evaluate [s]
    release : bool
        Whether to release the cloud (if not already released)

    Returns
    -------
    FreeFallSimulation
        Simulation results

    Examples
    --------
    >>> cloud = ThermalCloud(1.2e5, 295e-9, STATE_F2_MF2, position=[0, 0, 140e-6])
    >>> times = np.linspace(0, 20e-3, 100)
    >>> result = simulate_free_fall(cloud, times)
    >>> print(result.to_dataframe())
    """
    times = np.asarray(times)

    if release and not cloud.is_released:
        cloud.release()

    # For MixedStateCloud, use the first cloud (they all have same center position)
    if isinstance(cloud, MixedStateCloud):
        reference_cloud = cloud.clouds[0]
    else:
        reference_cloud = cloud

    # Evolve and collect states
    n_times = len(times)
    positions = np.zeros((n_times, 3))
    velocities = np.zeros((n_times, 3))
    sigmas = np.zeros((n_times, 3))
    states = []

    for i, t in enumerate(times):
        if isinstance(cloud, MixedStateCloud):
            cloud_states = cloud.evolve(t)
            state = cloud_states[0]  # Use first state for position
            # Average sigma across states (weighted by atom number)
            total_atoms = sum(c.n_atoms for c in cloud.clouds)
            avg_sigma = sum(
                c.sigma * c.n_atoms for c in cloud.clouds
            ) / total_atoms
            state = CloudState(
                time=t,
                position=state.position,
                velocity=state.velocity,
                sigma=avg_sigma,
                n_atoms=total_atoms,
                temperature=cloud.temperature
            )
        else:
            state = cloud.evolve(t)

        positions[i] = state.position
        velocities[i] = state.velocity
        sigmas[i] = state.sigma
        states.append(state)

    # Store parameters
    params = {
        'n_atoms': cloud.n_atoms_total if isinstance(cloud, MixedStateCloud) else cloud.n_atoms,
        'temperature_nK': cloud.temperature * 1e9,
        'initial_position_um': (cloud.clouds[0].initial_position * 1e6).tolist()
        if isinstance(cloud, MixedStateCloud) else (cloud.initial_position * 1e6).tolist(),
    }

    return FreeFallSimulation(
        times=times,
        positions=positions,
        velocities=velocities,
        sigmas=sigmas,
        states=states,
        params=params
    )


def create_experiment_cloud(
    params: Optional[ExperimentParams] = None,
    mixed_state: bool = True,
    random_temperature: bool = False
) -> Union[ThermalCloud, MixedStateCloud]:
    """
    Create a cloud with experimental parameters.

    Parameters
    ----------
    params : ExperimentParams, optional
        Experimental parameters. Uses defaults if not provided.
    mixed_state : bool
        If True, create MixedStateCloud with mF=2 and mF=1
    random_temperature : bool
        If True, randomize temperature within experimental range

    Returns
    -------
    ThermalCloud or MixedStateCloud
        Cloud configured with experimental parameters
    """
    if params is None:
        params = DEFAULT_PARAMS

    # Get temperature
    if random_temperature:
        temperature = params.get_random_temperature()
    else:
        temperature = params.temperature

    # Initial position: release from trap
    position = np.array([0.0, 0.0, params.release_height])

    if mixed_state:
        populations = {
            STATE_F2_MF2: params.population_mF2,
            STATE_F2_MF1: params.population_mF1
        }
        cloud = MixedStateCloud(
            n_atoms_total=params.n_atoms,
            temperature=temperature,
            populations=populations,
            position=position
        )
    else:
        cloud = ThermalCloud(
            n_atoms=params.n_atoms,
            temperature=temperature,
            state=STATE_F2_MF2,
            position=position
        )

    return cloud


def simulate_tof_series(
    observation_times: Optional[np.ndarray] = None,
    n_shots: int = 1,
    params: Optional[ExperimentParams] = None,
    random_fluctuations: bool = True
) -> List[FreeFallSimulation]:
    """
    Simulate a series of time-of-flight measurements.

    Like taking multiple shots in the experiment.

    Parameters
    ----------
    observation_times : array-like, optional
        Times to observe [s]. Default: experimental observation times
    n_shots : int
        Number of shots to simulate
    params : ExperimentParams, optional
        Experimental parameters
    random_fluctuations : bool
        Include random shot-to-shot fluctuations

    Returns
    -------
    list of FreeFallSimulation
        Results for each shot
    """
    if params is None:
        params = DEFAULT_PARAMS

    if observation_times is None:
        observation_times = np.array(params.observation_times)

    results = []

    for shot in range(n_shots):
        cloud = create_experiment_cloud(
            params=params,
            mixed_state=True,
            random_temperature=random_fluctuations
        )

        # Simulate with fine time resolution up to max observation time
        t_max = observation_times.max()
        times = np.linspace(0, t_max, 200)

        result = simulate_free_fall(cloud, times)
        results.append(result)

    return results


def calculate_collision_height(
    release_height: float,
    lobe_separation: float,
    v0: float = 0
) -> Tuple[float, float]:
    """
    Calculate when cloud reaches the Weber beam dark region.

    Parameters
    ----------
    release_height : float
        Initial height [m]
    lobe_separation : float
        Distance to beam center (half the dark region width) [m]
    v0 : float
        Initial velocity [m/s]

    Returns
    -------
    t_collision : float
        Time to reach beam [s]
    v_collision : float
        Velocity at collision [m/s]
    """
    # Solve: release_height - 0.5*g*t² = lobe_separation/2
    # 0.5*g*t² = release_height - lobe_separation/2
    # t = sqrt(2*(release_height - lobe_separation/2)/g)

    fall_distance = release_height - lobe_separation / 2

    if fall_distance <= 0:
        raise ValueError("Cloud starts below beam!")

    t_collision = np.sqrt(2 * fall_distance / G)
    v_collision = G * t_collision

    return t_collision, v_collision
