# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:35:55 2026

@author: clher
"""

"""
Cloud analysis routines.

Extract physical parameters from cloud images and simulations:
- Position (center of mass)
- Width/size (Gaussian sigma, FWHM)
- Atom number
- Temperature estimation
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from scipy.optimize import curve_fit

from .profiles import (
    Profile1D,
    extract_horizontal_profile,
    extract_vertical_profile,
    find_cloud_center,
    extract_profiles_at_center
)


@dataclass
class CloudParameters:
    """
    Extracted parameters of an atomic cloud.
    
    Attributes
    ----------
    center : tuple
        Center position (x, y) in specified units
    sigma : tuple
        Gaussian widths (σx, σy) in specified units
    amplitude : float
        Peak density
    offset : float
        Background offset
    n_atoms : float, optional
        Estimated atom number
    position_unit : str
        Unit of position/size values
    """
    center: Tuple[float, float]
    sigma: Tuple[float, float]
    amplitude: float
    offset: float
    n_atoms: Optional[float] = None
    position_unit: str = 'pixels'
    chi_squared: Optional[float] = None
    
    @property
    def fwhm(self) -> Tuple[float, float]:
        """Full width at half maximum."""
        return (2.355 * self.sigma[0], 2.355 * self.sigma[1])
    
    @property 
    def sigma_um(self) -> Tuple[float, float]:
        """Sigma in micrometers (if position is in meters)."""
        if self.position_unit == 'm':
            return (self.sigma[0] * 1e6, self.sigma[1] * 1e6)
        raise ValueError("Position unit is not meters")
    
    @property
    def center_um(self) -> Tuple[float, float]:
        """Center in micrometers."""
        if self.position_unit == 'm':
            return (self.center[0] * 1e6, self.center[1] * 1e6)
        raise ValueError("Position unit is not meters")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            'center_x': self.center[0],
            'center_y': self.center[1],
            'sigma_x': self.sigma[0],
            'sigma_y': self.sigma[1],
            'fwhm_x': self.fwhm[0],
            'fwhm_y': self.fwhm[1],
            'amplitude': self.amplitude,
            'offset': self.offset,
            'position_unit': self.position_unit,
        }
        if self.n_atoms is not None:
            result['n_atoms'] = self.n_atoms
        if self.chi_squared is not None:
            result['chi_squared'] = self.chi_squared
        return result
    
    def __repr__(self):
        unit = 'µm' if self.position_unit == 'm' else self.position_unit
        if self.position_unit == 'm':
            cx, cy = self.center_um
            sx, sy = self.sigma_um
        else:
            cx, cy = self.center
            sx, sy = self.sigma
        return (f"CloudParameters(center=({cx:.1f}, {cy:.1f}){unit}, "
                f"σ=({sx:.1f}, {sy:.1f}){unit})")


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def gaussian_1d(x, amplitude, center, sigma, offset):
    """1D Gaussian function."""
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian function (no rotation).
    
    Parameters
    ----------
    coords : tuple of arrays
        (x, y) coordinate arrays
    """
    x, y = coords
    return (offset + amplitude * np.exp(
        -(x - x0)**2 / (2 * sigma_x**2)
        -(y - y0)**2 / (2 * sigma_y**2)
    )).ravel()


def gaussian_2d_rotated(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function with rotation.
    
    Parameters
    ----------
    coords : tuple of arrays
        (x, y) coordinate arrays
    theta : float
        Rotation angle [radians]
    """
    x, y = coords
    
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = np.sin(2*theta) / (4*sigma_x**2) - np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    
    return (offset + amplitude * np.exp(
        -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)
    )).ravel()


# =============================================================================
# FITTING ROUTINES
# =============================================================================

def fit_gaussian_1d(
    x: np.ndarray,
    y: np.ndarray,
    p0: Optional[Tuple] = None
) -> Dict:
    """
    Fit 1D Gaussian to data.
    
    Parameters
    ----------
    x : array
        Position array
    y : array
        Data values
    p0 : tuple, optional
        Initial guess (amplitude, center, sigma, offset)
    
    Returns
    -------
    dict
        Fit results with keys: 'amplitude', 'center', 'sigma', 'offset',
        'covariance', 'residuals', 'chi_squared'
    """
    # Automatic initial guess
    if p0 is None:
        offset = np.median(y)
        amplitude = y.max() - offset
        center = x[np.argmax(y)]
        
        # Estimate sigma from FWHM
        half_max = amplitude / 2 + offset
        above_half = x[y > half_max]
        if len(above_half) > 1:
            sigma = (above_half.max() - above_half.min()) / 2.355
        else:
            sigma = (x.max() - x.min()) / 10
        
        p0 = (amplitude, center, max(sigma, 1e-10), offset)
    
    try:
        popt, pcov = curve_fit(
            gaussian_1d, x, y, p0=p0,
            bounds=(
                [0, x.min(), 0, -np.inf],  # Lower bounds
                [np.inf, x.max(), x.max()-x.min(), np.inf]  # Upper bounds
            ),
            maxfev=5000
        )
        
        residuals = y - gaussian_1d(x, *popt)
        chi_sq = np.sum(residuals**2) / (len(y) - len(popt))
        
        return {
            'amplitude': popt[0],
            'center': popt[1],
            'sigma': abs(popt[2]),
            'offset': popt[3],
            'covariance': pcov,
            'residuals': residuals,
            'chi_squared': chi_sq,
            'success': True
        }
        
    except (RuntimeError, ValueError) as e:
        return {
            'amplitude': p0[0],
            'center': p0[1],
            'sigma': p0[2],
            'offset': p0[3],
            'covariance': None,
            'residuals': None,
            'chi_squared': None,
            'success': False,
            'error': str(e)
        }


def fit_gaussian_2d(
    image: np.ndarray,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    rotated: bool = False
) -> Dict:
    """
    Fit 2D Gaussian to image.
    
    Parameters
    ----------
    image : 2D array
        Image data
    x, y : arrays, optional
        Coordinate arrays. If None, uses pixel indices.
    rotated : bool
        If True, fit with rotation angle
    
    Returns
    -------
    dict
        Fit results
    """
    ny, nx = image.shape
    
    if x is None:
        x = np.arange(nx)
    if y is None:
        y = np.arange(ny)
    
    X, Y = np.meshgrid(x, y)
    
    # Initial guess
    offset = np.median(image)
    amplitude = image.max() - offset
    
    # Find center
    cy_idx, cx_idx = np.unravel_index(np.argmax(image), image.shape)
    x0 = x[cx_idx] if cx_idx < len(x) else x[len(x)//2]
    y0 = y[cy_idx] if cy_idx < len(y) else y[len(y)//2]
    
    # Estimate sigma from profiles
    h_profile = image[cy_idx, :]
    v_profile = image[:, cx_idx]
    
    h_fit = fit_gaussian_1d(x, h_profile)
    v_fit = fit_gaussian_1d(y, v_profile)
    
    sigma_x = h_fit['sigma'] if h_fit['success'] else (x.max() - x.min()) / 10
    sigma_y = v_fit['sigma'] if v_fit['success'] else (y.max() - y.min()) / 10
    
    if rotated:
        p0 = (amplitude, x0, y0, sigma_x, sigma_y, 0, offset)
        fit_func = gaussian_2d_rotated
        param_names = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
    else:
        p0 = (amplitude, x0, y0, sigma_x, sigma_y, offset)
        fit_func = gaussian_2d
        param_names = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'offset']
    
    try:
        popt, pcov = curve_fit(
            fit_func,
            (X, Y),
            image.ravel(),
            p0=p0,
            maxfev=10000
        )
        
        fitted_image = fit_func((X, Y), *popt).reshape(image.shape)
        residuals = image - fitted_image
        chi_sq = np.sum(residuals**2) / (image.size - len(popt))
        
        result = {name: val for name, val in zip(param_names, popt)}
        result['sigma_x'] = abs(result['sigma_x'])
        result['sigma_y'] = abs(result['sigma_y'])
        result['covariance'] = pcov
        result['residuals'] = residuals
        result['chi_squared'] = chi_sq
        result['success'] = True
        
        return result
        
    except (RuntimeError, ValueError) as e:
        result = {name: val for name, val in zip(param_names, p0)}
        result['covariance'] = None
        result['residuals'] = None
        result['chi_squared'] = None
        result['success'] = False
        result['error'] = str(e)
        return result


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_cloud_image(
    image: np.ndarray,
    pixel_size: Optional[float] = None,
    method: str = '2d',
    average_width: int = 5
) -> CloudParameters:
    """
    Analyze cloud image and extract parameters.
    
    Parameters
    ----------
    image : 2D array
        Cloud image
    pixel_size : float, optional
        Pixel size in meters
    method : str
        '2d' for full 2D Gaussian fit, '1d' for separate 1D fits
    average_width : int
        Width for averaging profiles (for 1d method)
    
    Returns
    -------
    CloudParameters
        Extracted cloud parameters
    """
    ny, nx = image.shape
    
    if pixel_size is not None:
        x = (np.arange(nx) - nx/2) * pixel_size
        y = (np.arange(ny) - ny/2) * pixel_size
        position_unit = 'm'
    else:
        x = np.arange(nx)
        y = np.arange(ny)
        position_unit = 'pixels'
    
    if method == '2d':
        fit_result = fit_gaussian_2d(image, x, y)
        
        return CloudParameters(
            center=(fit_result['x0'], fit_result['y0']),
            sigma=(fit_result['sigma_x'], fit_result['sigma_y']),
            amplitude=fit_result['amplitude'],
            offset=fit_result['offset'],
            position_unit=position_unit,
            chi_squared=fit_result['chi_squared']
        )
    
    elif method == '1d':
        h_profile, v_profile = extract_profiles_at_center(
            image, average_width=average_width, pixel_size=pixel_size
        )
        
        h_fit = fit_gaussian_1d(h_profile.position, h_profile.values)
        v_fit = fit_gaussian_1d(v_profile.position, v_profile.values)
        
        return CloudParameters(
            center=(h_fit['center'], v_fit['center']),
            sigma=(h_fit['sigma'], v_fit['sigma']),
            amplitude=(h_fit['amplitude'] + v_fit['amplitude']) / 2,
            offset=(h_fit['offset'] + v_fit['offset']) / 2,
            position_unit=position_unit
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_cloud_series(
    images: List[np.ndarray],
    times: np.ndarray,
    pixel_size: Optional[float] = None,
    method: str = '2d'
) -> Dict:
    """
    Analyze a series of cloud images at different times.
    
    Useful for time-of-flight analysis.
    
    Parameters
    ----------
    images : list of 2D arrays
        Cloud images at different times
    times : array
        Time of each image [s]
    pixel_size : float, optional
        Pixel size in meters
    method : str
        Fitting method ('2d' or '1d')
    
    Returns
    -------
    dict
        Dictionary with arrays of extracted parameters vs time
    """
    n_images = len(images)
    
    centers_x = np.zeros(n_images)
    centers_y = np.zeros(n_images)
    sigmas_x = np.zeros(n_images)
    sigmas_y = np.zeros(n_images)
    amplitudes = np.zeros(n_images)
    
    for i, img in enumerate(images):
        params = analyze_cloud_image(img, pixel_size=pixel_size, method=method)
        centers_x[i] = params.center[0]
        centers_y[i] = params.center[1]
        sigmas_x[i] = params.sigma[0]
        sigmas_y[i] = params.sigma[1]
        amplitudes[i] = params.amplitude
    
    return {
        'times': times,
        'center_x': centers_x,
        'center_y': centers_y,
        'sigma_x': sigmas_x,
        'sigma_y': sigmas_y,
        'amplitude': amplitudes,
        'pixel_size': pixel_size
    }


def estimate_temperature_from_tof(
    sigmas: np.ndarray,
    times: np.ndarray,
    trap_frequency: float,
    mass: float = 1.443160648e-25  # Rb-87
) -> Tuple[float, float]:
    """
    Estimate temperature from time-of-flight expansion.
    
    Uses: σ²(t) = σ₀² + (k_B T / m) × t²
    
    Parameters
    ----------
    sigmas : array
        Cloud sizes at different times [m]
    times : array
        Time points [s]
    trap_frequency : float
        Initial trap frequency [rad/s]
    mass : float
        Atomic mass [kg]
    
    Returns
    -------
    temperature : float
        Estimated temperature [K]
    sigma_0 : float
        Initial cloud size [m]
    """
    from ..utils.constants import KB
    
    # Fit σ²(t) = σ₀² + v_th² × t²
    # which is σ²(t) = σ₀² + (k_B T / m) × t²
    
    sigma_squared = sigmas**2
    t_squared = times**2
    
    # Linear fit: σ² = a + b × t²
    # where a = σ₀², b = k_B T / m
    A = np.column_stack([np.ones_like(t_squared), t_squared])
    result, residuals, rank, s = np.linalg.lstsq(A, sigma_squared, rcond=None)
    
    sigma_0_squared = result[0]
    kb_T_over_m = result[1]
    
    sigma_0 = np.sqrt(max(sigma_0_squared, 0))
    temperature = kb_T_over_m * mass / KB
    
    return temperature, sigma_0


def extract_position_vs_time(
    simulation_result,
    direction: str = 'z'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract position along one direction from simulation result.
    
    Parameters
    ----------
    simulation_result : FreeFallSimulation
        Result from simulate_free_fall
    direction : str
        'x', 'y', or 'z'
    
    Returns
    -------
    times : array
        Time array [s]
    positions : array
        Position array [m]
    """
    idx = {'x': 0, 'y': 1, 'z': 2}[direction]
    return simulation_result.times, simulation_result.positions[:, idx]


def extract_size_vs_time(
    simulation_result,
    direction: str = 'radial'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract cloud size along one direction from simulation result.
    
    Parameters
    ----------
    simulation_result : FreeFallSimulation
        Result from simulate_free_fall
    direction : str
        'x', 'y', 'z', 'radial' (average of x,y), or 'all'
    
    Returns
    -------
    times : array
        Time array [s]
    sizes : array
        Size array [m] (or dict if direction='all')
    """
    times = simulation_result.times
    sigmas = simulation_result.sigmas
    
    if direction == 'x':
        return times, sigmas[:, 0]
    elif direction == 'y':
        return times, sigmas[:, 1]
    elif direction == 'z':
        return times, sigmas[:, 2]
    elif direction == 'radial':
        return times, (sigmas[:, 0] + sigmas[:, 1]) / 2
    elif direction == 'all':
        return times, {
            'x': sigmas[:, 0],
            'y': sigmas[:, 1],
            'z': sigmas[:, 2],
            'radial': (sigmas[:, 0] + sigmas[:, 1]) / 2
        }
    else:
        raise ValueError(f"Unknown direction: {direction}")