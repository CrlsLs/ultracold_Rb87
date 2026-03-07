# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:22:12 2026

@author: clher
"""
"""Ultracold cloud simulation module."""




import numpy as np
from ..utils.constants import HBAR, KB, M_RB87
class UltracoldCloud:
    """
    Represents an ultracold atomic cloud.

    Parameters
    ----------
    n_atoms : float
        Number of atoms in the cloud
    temperature : float
        Temperature in Kelvin
    trap_frequencies : tuple
        Trap frequencies (omega_x, omega_y, omega_z) in rad/s
    """

    def __init__(self, n_atoms, temperature, trap_frequencies):
        self.n_atoms = n_atoms
        self.temperature = temperature
        self.omega_x, self.omega_y, self.omega_z = trap_frequencies
        self.mass = M_RB87

    @property
    def thermal_sizes(self):
        """Calculate thermal cloud sizes (sigma_x, sigma_y, sigma_z)."""
        sigma_x = np.sqrt(KB * self.temperature /
                          (self.mass * self.omega_x**2))
        sigma_y = np.sqrt(KB * self.temperature /
                          (self.mass * self.omega_y**2))
        sigma_z = np.sqrt(KB * self.temperature /
                          (self.mass * self.omega_z**2))
        return sigma_x, sigma_y, sigma_z

    def density_distribution(self, x, y, z=0):
        """
        Calculate 3D Gaussian density distribution.

        Returns density in atoms/m³
        """
        sigma_x, sigma_y, sigma_z = self.thermal_sizes

        norm = self.n_atoms / ((2 * np.pi)**1.5 * sigma_x * sigma_y * sigma_z)

        density = norm * np.exp(
            -x**2 / (2 * sigma_x**2)
            - y**2 / (2 * sigma_y**2)
            - z**2 / (2 * sigma_z**2)
        )
        return density

    def column_density(self, x, y):
        """
        Calculate column density (integrated along z).

        Returns density in atoms/m²
        """
        sigma_x, sigma_y, _ = self.thermal_sizes

        norm = self.n_atoms / (2 * np.pi * sigma_x * sigma_y)

        return norm * np.exp(-x**2 / (2 * sigma_x**2) - y**2 / (2 * sigma_y**2))

    def generate_image(self, x_grid, y_grid, noise_level=0.01):
        """Generate a simulated absorption image."""
        density = self.column_density(x_grid, y_grid)

        # Add realistic noise
        noise = np.random.normal(0, noise_level * density.max(), density.shape)
        return density + noise


class BECCloud(UltracoldCloud):
    """
    Bose-Einstein Condensate with Thomas-Fermi profile.
    """

    def __init__(self, n_atoms, chemical_potential, trap_frequencies):
        super().__init__(n_atoms, 0, trap_frequencies)  # T=0 for pure BEC
        self.mu = chemical_potential

    @property
    def thomas_fermi_radii(self):
        """Calculate Thomas-Fermi radii."""
        r_x = np.sqrt(2 * self.mu / (self.mass * self.omega_x**2))
        r_y = np.sqrt(2 * self.mu / (self.mass * self.omega_y**2))
        r_z = np.sqrt(2 * self.mu / (self.mass * self.omega_z**2))
        return r_x, r_y, r_z

    def column_density(self, x, y):
        """Thomas-Fermi column density profile."""
        r_x, r_y, r_z = self.thomas_fermi_radii

        # Parabolic profile
        arg = 1 - (x/r_x)**2 - (y/r_y)**2
        arg = np.maximum(arg, 0)  # Set negative values to zero

        # Normalization for TF profile
        norm = 15 * self.n_atoms / (8 * np.pi * r_x * r_y)

        return norm * arg**1.5
