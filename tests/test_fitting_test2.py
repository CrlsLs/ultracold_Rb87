# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:31:19 2026

@author: clher
"""
"""Tests for fitting module."""


class TestGaussianFit:
    """Tests for Gaussian fitting."""

    def test_perfect_gaussian(self):
        """Fit should recover exact parameters for noiseless data."""
        x = np.linspace(-10, 10, 200)
        true_params = {'amplitude': 5.0,
                       'center': 1.0, 'sigma': 2.0, 'offset': 0.5}
        y = gaussian_1d(x, **true_params)

        result = fit_gaussian_1d(x, y)

        assert abs(result.parameters['amplitude'] - 5.0) < 0.01
        assert abs(result.parameters['center'] - 1.0) < 0.01
        assert abs(result.parameters['sigma'] - 2.0) < 0.01
        assert abs(result.parameters['offset'] - 0.5) < 0.01

    def test_noisy_gaussian(self):
        """Fit should work with moderate noise."""
        np.random.seed(42)
        x = np.linspace(-10, 10, 200)
        true_params = {'amplitude': 10.0,
                       'center': 0.0, 'sigma': 3.0, 'offset': 1.0}
        y = gaussian_1d(x, **true_params) + np.random.normal(0, 0.5, len(x))

        result = fit_gaussian_1d(x, y)

        # Allow larger tolerance for noisy data
        assert abs(result.parameters['amplitude'] - 10.0) < 1.0
        assert abs(result.parameters['center'] - 0.0) < 0.5
        assert abs(result.parameters['sigma'] - 3.0) < 0.5

    def test_fit_result_has_chi_squared(self):
        """FitResult should include chi-squared."""
        x = np.linspace(-5, 5, 100)
        y = gaussian_1d(x, 1, 0, 1, 0)

        result = fit_gaussian_1d(x, y)

        assert hasattr(result, 'chi_squared')
        assert result.chi_squared >= 0


class TestThomasFermi:
    """Tests for Thomas-Fermi profile."""

    def test_tf_zero_outside_radius(self):
        """TF profile should be zero outside the radius."""
        x = np.array([0, 5, 10, 15, 20])
        y = thomas_fermi_1d(x, amplitude=10, center=10, radius=5, offset=0)

        assert y[0] == 0  # x=0, outside radius
        assert y[-1] == 0  # x=20, outside radius
        assert y[2] > 0    # x=10, at center

    def test_tf_maximum_at_center(self):
        """TF profile maximum should be at center."""
        x = np.linspace(0, 20, 100)
        y = thomas_fermi_1d(x, amplitude=10, center=10, radius=5, offset=0)

        assert np.argmax(y) == np.argmin(np.abs(x - 10))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
