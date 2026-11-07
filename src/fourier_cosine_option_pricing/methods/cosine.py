"""Fourier-Cosine pricing for European call and put options."""

import numpy as np


class FourierCosineMethod:
    """Price a European option from a model's characteristic function."""

    def __init__(self, model, spot, strike, grid_size=256, call=True):
        if spot <= 0 or strike <= 0:
            raise ValueError("spot and strike must be positive")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        self.model = model
        self.spot = float(spot)
        self.strike = float(strike)
        self.grid_size = int(grid_size)
        self.call = bool(call)
        self.log_moneyness = np.log(self.spot / self.strike)
        self.a, self.b = self.model.integral_truncation()

    @staticmethod
    def _payoff_coefficients(k, a, lower, upper):
        """COS coefficients of max(exp(x)-1, 0) on [lower, upper]."""
        length = upper - a
        omega = k * np.pi / length
        xhi = (
            np.exp(upper) * (np.cos(omega * (upper - a)) + omega * np.sin(omega * (upper - a)))
            - np.exp(lower) * (np.cos(omega * (lower - a)) + omega * np.sin(omega * (lower - a)))
        ) / (1 + omega**2)
        psi = np.empty_like(k, dtype=float)
        psi[0] = upper - lower
        psi[1:] = (np.sin(omega[1:] * (upper - a)) - np.sin(omega[1:] * (lower - a))) / omega[1:]
        return xhi - psi

    def price(self):
        """Compute and return the discounted option price."""
        # Expand the terminal log-moneyness X = log(S_T/K), whose bounds are
        # the log-return bounds shifted by the initial log-moneyness.
        a = self.a + self.log_moneyness
        b = self.b + self.log_moneyness
        k = np.arange(self.grid_size, dtype=float)

        if self.call:
            lower, upper = max(0.0, a), b
            sign = 1.0
        else:
            lower, upper = a, min(0.0, b)
            sign = -1.0
        if lower >= upper:
            return 0.0

        payoff = self._payoff_coefficients(k, a, lower, upper)
        u = k * np.pi / (b - a)
        characteristic = self.model.characteristic_function(u) * np.exp(1j * u * self.log_moneyness)
        weights = np.ones(self.grid_size)
        weights[0] = 0.5
        discounted = np.exp(-self.model.rate * self.model.maturity)
        value = sign * np.real(np.dot(weights, characteristic * np.exp(-1j * u * a) * payoff))
        return float(self.strike * discounted * (2.0 / (b - a)) * value)
