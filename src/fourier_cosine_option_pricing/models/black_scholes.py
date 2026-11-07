"""Black-Scholes (geometric Brownian motion) model."""

import numpy as np


class BlackScholesModel:
    """Log-return model with constant volatility and risk-free rate."""

    def __init__(self, sigma: float, rate: float, maturity: float):
        if sigma <= 0 or maturity <= 0:
            raise ValueError("sigma and maturity must be positive")
        self.sigma = float(sigma)
        self.rate = float(rate)
        self.maturity = float(maturity)
        # Legacy attribute aliases.
        self.rf = self.rate
        self.mat = self.maturity

    def characteristic_function(self, grid):
        """Characteristic function of the log return over the maturity."""
        u = np.asarray(grid, dtype=complex)
        drift = (self.rate - 0.5 * self.sigma**2) * self.maturity
        variance = self.sigma**2 * self.maturity
        return np.exp(1j * u * drift - 0.5 * variance * u**2)

    def integral_truncation(self):
        """Return a cumulant-based integration interval for log returns."""
        L = 10.0
        drift = (self.rate - 0.5 * self.sigma**2) * self.maturity
        width = L * self.sigma * np.sqrt(self.maturity)
        return drift - width, drift + width
