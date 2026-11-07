"""Carr-Madan Fourier pricing helper, retained from the original source."""

import numpy as np
from scipy import integrate


class FourierCarrMadanMethod:
    """Carr-Madan damped Fourier integral for a Heston call option."""

    @staticmethod
    def call_price_heston(S, K, T, r, kappa, theta, nu, rho, V_0, alpha=1, L=1000):
        if S <= 0 or K <= 0 or T <= 0 or nu == 0:
            raise ValueError("S, K, T must be positive and nu must be non-zero")
        b = lambda x: kappa - 1j * rho * nu * x
        gamma = lambda x: np.sqrt(nu**2 * (x**2 + 1j * x) + b(x) ** 2)
        a = lambda x: (b(x) / gamma(x)) * np.sinh(T * 0.5 * gamma(x))
        c = lambda x: (gamma(x) * np.cosh(0.5 * T * gamma(x))) / np.sinh(T * 0.5 * gamma(x) + b(x))
        d = lambda x: kappa * theta * T * b(x) / nu**2
        f = lambda x: 1j * (np.log(S) + r * T) * x + d(x)
        g = lambda x: (np.cosh(T * 0.5 * gamma(x)) + a(x)) ** (2 * kappa * theta / nu**2)
        h = lambda x: -(x**2 + 1j * x) * V_0 / c(x)
        phi = lambda x: np.exp(f(x)) * np.exp(h(x) / g(x))
        integrand = lambda x: np.real(phi(x - 1j * (alpha + 1)) / ((alpha + 1j * x) * (alpha + 1 + 1j * x))) * np.exp(-1j * np.log(K) * x)
        integral = integrate.quad(integrand, 0, L)
        return np.exp(-r * T - alpha * np.log(K)) * integral[0] / np.pi
