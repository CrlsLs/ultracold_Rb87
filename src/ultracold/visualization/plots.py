# -*- coding: utf-8 -*-

"""Profile extraction from 2D images."""

import numpy as np
from typing import Tuple, Optional


def extract_horizontal_profile(image: np.ndarray,
                               row: Optional[int] = None,
                               average_rows: int = 1) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    x : array
        Pixel positions
    profile : array
        Intensity profile
    """
    if row is None:
        # Find row with maximum signal
        row_sums = np.sum(image, axis=1)
        row = np.argmax(row_sums)

    half_avg = average_rows // 2
    row_start = max(0, row - half_avg)
    row_end = min(image.shape[0], row + half_avg + 1)

    profile = np.mean(image[row_start:row_end, :], axis=0)
    x = np.arange(len(profile))

    return x, profile


def extract_vertical_profile(image: np.ndarray,
                             col: Optional[int] = None,
                             average_cols: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vertical profile from image."""
    if col is None:
        col_sums = np.sum(image, axis=0)
        col = np.argmax(col_sums)

    half_avg = average_cols // 2
    col_start = max(0, col - half_avg)
    col_end = min(image.shape[1], col + half_avg + 1)

    profile = np.mean(image[:, col_start:col_end], axis=1)
    y = np.arange(len(profile))

    return y, profile


def extract_radial_profile(image: np.ndarray,
                           center: Optional[Tuple[int, int]] = None,
                           n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract azimuthally averaged radial profile.

    Returns
    -------
    r : array
        Radial positions
    profile : array
        Averaged intensity
    std : array
        Standard deviation at each radius
    """
    if center is None:
        # Find center of mass
        y_idx, x_idx = np.indices(image.shape)
        total = image.sum()
        cx = (x_idx * image).sum() / total
        cy = (y_idx * image).sum() / total
        center = (int(cy), int(cx))

    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

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

    return r_centers, profile, std
