# -*- coding: utf-8 -*-

"""Profile extraction from 2D images."""
"""
Profile extraction from 2D images.

Functions for extracting 1D profiles, radial profiles,
and analyzing spatial distributions of atomic clouds.
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class Profile1D:
    """
    Container for 1D profile data.
    
    Attributes
    ----------
    position : ndarray
        Position coordinates [m] or [pixels]
    values : ndarray
        Profile values
    position_unit : str
        Unit of position ('m', 'um', 'pixels')
    direction : str
        Direction of profile ('horizontal', 'vertical', 'radial')
    center_index : int, optional
        Index of profile center
    """
    position: np.ndarray
    values: np.ndarray
    position_unit: str = 'pixels'
    direction: str = 'horizontal'
    center_index: Optional[int] = None
    
    @property
    def position_um(self) -> np.ndarray:
        """Position in micrometers (if available)."""
        if self.position_unit == 'm':
            return self.position * 1e6
        elif self.position_unit == 'um':
            return self.position
        else:
            raise ValueError("Position not in metric units")
    
    @property
    def peak_value(self) -> float:
        """Maximum value."""
        return np.max(self.values)
    
    @property
    def peak_position(self) -> float:
        """Position of maximum."""
        return self.position[np.argmax(self.values)]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'position': self.position.tolist(),
            'values': self.values.tolist(),
            'position_unit': self.position_unit,
            'direction': self.direction,
            'peak_value': float(self.peak_value),
            'peak_position': float(self.peak_position)
        }


def extract_horizontal_profile(
    image: np.ndarray,
    row: Optional[int] = None,
    average_rows: int = 1,
    pixel_size: Optional[float] = None
) -> Profile1D:
    """
    Extract horizontal profile from image.
    
    Parameters
    ----------
    image : 2D array
        Input image
    row : int, optional
        Row index. If None, uses row with maximum signal.
    average_rows : int
        Number of rows to average around the center row
    pixel_size : float, optional
        Pixel size in meters. If provided, position is in meters.
    
    Returns
    -------
    Profile1D
        Extracted profile with position and values
    
    Examples
    --------
    >>> profile = extract_horizontal_profile(image, average_rows=5)
    >>> plt.plot(profile.position, profile.values)
    """
    if row is None:
        # Find row with maximum signal
        row_sums = np.sum(image, axis=1)
        row = int(np.argmax(row_sums))
    
    half_avg = average_rows // 2
    row_start = max(0, row - half_avg)
    row_end = min(image.shape[0], row + half_avg + 1)
    
    values = np.mean(image[row_start:row_end, :], axis=0)
    
    # Position array
    n_cols = image.shape[1]
    if pixel_size is not None:
        # Center at middle of image
        position = (np.arange(n_cols) - n_cols/2) * pixel_size
        position_unit = 'm'
    else:
        position = np.arange(n_cols)
        position_unit = 'pixels'
    
    return Profile1D(
        position=position,
        values=values,
        position_unit=position_unit,
        direction='horizontal',
        center_index=row
    )


def extract_vertical_profile(
    image: np.ndarray,
    col: Optional[int] = None,
    average_cols: int = 1,
    pixel_size: Optional[float] = None
) -> Profile1D:
    """
    Extract vertical profile from image.
    
    Parameters
    ----------
    image : 2D array
        Input image
    col : int, optional
        Column index. If None, uses column with maximum signal.
    average_cols : int
        Number of columns to average around the center column
    pixel_size : float, optional
        Pixel size in meters
    
    Returns
    -------
    Profile1D
        Extracted profile
    """
    if col is None:
        col_sums = np.sum(image, axis=0)
        col = int(np.argmax(col_sums))
    
    half_avg = average_cols // 2
    col_start = max(0, col - half_avg)
    col_end = min(image.shape[1], col + half_avg + 1)
    
    values = np.mean(image[:, col_start:col_end], axis=1)
    
    n_rows = image.shape[0]
    if pixel_size is not None:
        position = (np.arange(n_rows) - n_rows/2) * pixel_size
        position_unit = 'm'
    else:
        position = np.arange(n_rows)
        position_unit = 'pixels'
    
    return Profile1D(
        position=position,
        values=values,
        position_unit=position_unit,
        direction='vertical',
        center_index=col
    )


def extract_radial_profile(
    image: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    n_bins: int = 100,
    pixel_size: Optional[float] = None
) -> Tuple[Profile1D, np.ndarray]:
    """
    Extract azimuthally averaged radial profile.
    
    Parameters
    ----------
    image : 2D array
        Input image
    center : tuple of int, optional
        Center (row, col). If None, uses center of mass.
    n_bins : int
        Number of radial bins
    pixel_size : float, optional
        Pixel size in meters
    
    Returns
    -------
    Profile1D
        Radial profile
    std : ndarray
        Standard deviation at each radius
    """
    if center is None:
        # Find center of mass
        y_idx, x_idx = np.indices(image.shape)
        total = image.sum()
        if total > 0:
            cx = (x_idx * image).sum() / total
            cy = (y_idx * image).sum() / total
        else:
            cy, cx = image.shape[0]/2, image.shape[1]/2
        center = (int(cy), int(cx))
    
    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Maximum radius that fits in all directions
    r_max = min(center[0], center[1],
                image.shape[0] - center[0],
                image.shape[1] - center[1])
    
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    profile = np.zeros(n_bins)
    std = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if mask.sum() > 0:
            profile[i] = np.mean(image[mask])
            std[i] = np.std(image[mask])
    
    if pixel_size is not None:
        position = r_centers * pixel_size
        position_unit = 'm'
    else:
        position = r_centers
        position_unit = 'pixels'
    
    profile_obj = Profile1D(
        position=position,
        values=profile,
        position_unit=position_unit,
        direction='radial',
        center_index=None
    )
    
    return profile_obj, std


def find_cloud_center(
    image: np.ndarray,
    method: str = 'com'
) -> Tuple[float, float]:
    """
    Find the center of the atomic cloud.
    
    Parameters
    ----------
    image : 2D array
        Input image
    method : str
        Method: 'com' (center of mass), 'max' (maximum), 'fit' (Gaussian fit)
    
    Returns
    -------
    center_row, center_col : float
        Center coordinates
    """
    if method == 'com':
        y_idx, x_idx = np.indices(image.shape)
        total = image.sum()
        if total > 0:
            cx = (x_idx * image).sum() / total
            cy = (y_idx * image).sum() / total
        else:
            cy, cx = image.shape[0]/2, image.shape[1]/2
        return float(cy), float(cx)
    
    elif method == 'max':
        max_idx = np.unravel_index(np.argmax(image), image.shape)
        return float(max_idx[0]), float(max_idx[1])
    
    elif method == 'fit':
        # Simple parabolic fit around maximum
        max_row, max_col = find_cloud_center(image, method='max')
        
        # Refine with parabolic interpolation
        if 1 <= max_row < image.shape[0]-1:
            y_m1 = image[int(max_row)-1, int(max_col)]
            y_0 = image[int(max_row), int(max_col)]
            y_p1 = image[int(max_row)+1, int(max_col)]
            if y_m1 + y_p1 - 2*y_0 != 0:
                max_row += 0.5 * (y_m1 - y_p1) / (y_m1 + y_p1 - 2*y_0)
        
        if 1 <= max_col < image.shape[1]-1:
            x_m1 = image[int(max_row), int(max_col)-1]
            x_0 = image[int(max_row), int(max_col)]
            x_p1 = image[int(max_row), int(max_col)+1]
            if x_m1 + x_p1 - 2*x_0 != 0:
                max_col += 0.5 * (x_m1 - x_p1) / (x_m1 + x_p1 - 2*x_0)
        
        return float(max_row), float(max_col)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_profiles_at_center(
    image: np.ndarray,
    average_width: int = 5,
    pixel_size: Optional[float] = None
) -> Tuple[Profile1D, Profile1D]:
    """
    Extract horizontal and vertical profiles through the cloud center.
    
    Parameters
    ----------
    image : 2D array
        Input image
    average_width : int
        Number of rows/columns to average
    pixel_size : float, optional
        Pixel size in meters
    
    Returns
    -------
    h_profile, v_profile : Profile1D
        Horizontal and vertical profiles
    """
    center_row, center_col = find_cloud_center(image)
    
    h_profile = extract_horizontal_profile(
        image,
        row=int(center_row),
        average_rows=average_width,
        pixel_size=pixel_size
    )
    
    v_profile = extract_vertical_profile(
        image,
        col=int(center_col),
        average_cols=average_width,
        pixel_size=pixel_size
    )
    
    return h_profile, v_profile


def profile_from_simulation(
    cloud,
    direction: str,
    n_points: int = 200,
    extent: float = 500e-6
) -> Profile1D:
    """
    Extract profile directly from cloud object (no image generation).
    
    Parameters
    ----------
    cloud : ThermalCloud or MixedStateCloud
        Cloud object
    direction : str
        'x', 'y', or 'z'
    n_points : int
        Number of points in profile
    extent : float
        Half-extent of profile [m]
    
    Returns
    -------
    Profile1D
        1D density profile
    """
    position = np.linspace(-extent, extent, n_points)
    
    # Get cloud parameters
    if hasattr(cloud, 'clouds'):  # MixedStateCloud
        # Combine contributions
        values = np.zeros(n_points)
        for c in cloud.clouds:
            sigma = c.sigma
            pos = c.position
            n = c.n_atoms
            
            idx = {'x': 0, 'y': 1, 'z': 2}[direction]
            sigma_dir = sigma[idx]
            pos_dir = pos[idx]
            
            # 1D Gaussian (integrated column density)
            norm = n / (np.sqrt(2*np.pi) * sigma_dir)
            values += norm * np.exp(-(position - pos_dir)**2 / (2*sigma_dir**2))
    else:
        sigma = cloud.sigma
        pos = cloud.position
        n = cloud.n_atoms
        
        idx = {'x': 0, 'y': 1, 'z': 2}[direction]
        sigma_dir = sigma[idx]
        pos_dir = pos[idx]
        
        norm = n / (np.sqrt(2*np.pi) * sigma_dir)
        values = norm * np.exp(-(position - pos_dir)**2 / (2*sigma_dir**2))
    
    direction_map = {'x': 'horizontal', 'y': 'vertical', 'z': 'longitudinal'}
    
    return Profile1D(
        position=position,
        values=values,
        position_unit='m',
        direction=direction_map.get(direction, direction)
    )