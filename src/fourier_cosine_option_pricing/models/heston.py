"""Heston stochastic-volatility model."""

import numpy as np


class HestonModel:
    """Heston model using the original project's variance parameter names."""

    def __init__(self, kappa, theta, rho, eta, rf, sigma, mat):
        if kappa <= 0 or eta <= 0 or mat <= 0:
            raise ValueError("kappa, eta, and mat must be positive")
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.rho = float(rho)
        self.eta = float(eta)
        self.rf = float(rf)
        self.sigma = float(sigma)  # Initial variance, as in the original implementation.
        self.mat = float(mat)
        self.rate = self.rf
        self.maturity = self.mat

    def characteristic_function(self, grid):
        """Characteristic function of the Heston log return."""
        u = np.asarray(grid, dtype=complex)
        kappa, theta, v0 = self.kappa, self.theta, self.sigma
        rho, eta, rate, maturity = self.rho, self.eta, self.rf, self.mat

        d = np.sqrt((kappa - 1j * rho * eta * u) ** 2 + (u**2 + 1j * u) * eta**2)
        g = (kappa - 1j * rho * eta * u - d) / (kappa - 1j * rho * eta * u + d)
        exp_d = np.exp(-d * maturity)
        c = (1 - g * exp_d) / (1 - g)
        first = np.exp(
            1j * u * rate * maturity
            + (v0 / eta**2) * (kappa - 1j * rho * eta * u - d) * (1 - exp_d) / (1 - g * exp_d)
        )
        second = np.exp(
            (kappa * theta / eta**2)
            * ((kappa - 1j * rho * eta * u - d) * maturity - 2 * np.log(c))
        )
        return first * second

    def integral_truncation(self):
        """Approximate integration bounds from the first two log-return cumulants."""
        variance_scale = max(self.sigma, self.theta, 1e-12)
        center = (self.rf - 0.5 * variance_scale) * self.mat
        width = 12.0 * np.sqrt(variance_scale * self.mat)
        return center - width, center + width
