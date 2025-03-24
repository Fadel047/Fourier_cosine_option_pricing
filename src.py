import numpy as np

class SBM:
    """
    Standard Brownian Motion (GBM under log transformation)
    """
    def __init__(self, sigma, r, T):
        self.sigma = sigma
        self.r = r
        self.T = T

    def characteristic_function(self, u):
        """
        Characteristic function for GBM log-price
        """
        return np.exp(1j * u * self.r * self.T - 0.5 * (u ** 2) * self.sigma**2 * self.T)

    def truncation_range(self, L=10):
        """
        Truncation interval [a, b] based on cumulants (eq 26 in paper)
        """
        c1 = self.r * self.T
        c2 = self.sigma**2 * self.T
        a = c1 - L * np.sqrt(c2)
        b = c1 + L * np.sqrt(c2)
        return a, b


class FourierCosineMethod:
    def __init__(self, model, S0, K, N, call=True):
        self.model = model
        self.S0 = S0
        self.K = K
        self.N = N
        self.call = call
        self.x = np.log(S0 / K)
        self.a, self.b = model.truncation_range()

    def _chi_psi(self, k, a, b, c, d):
        """
        Computes chi and psi coefficients for COS expansion (eq 24, 25)
        """
        k_pi = k * np.pi / (b - a)
        chi = (np.cos(k_pi * (d - a)) * np.exp(d) - np.cos(k_pi * (c - a)) * np.exp(c) +
               k_pi * (np.sin(k_pi * (d - a)) * np.exp(d) - np.sin(k_pi * (c - a)) * np.exp(c))) / \
              (1 + k_pi**2)
        psi = np.where(k == 0, d - c,
                       (np.sin(k_pi * (d - a)) - np.sin(k_pi * (c - a))) / k_pi)
        return chi, psi

    def price(self):
        k = np.arange(self.N)
        u = k * np.pi / (self.b - self.a)
        phi = self.model.characteristic_function(u)

        # payoff coefficients
        if self.call:
            c, d = 0.0, self.b
            chi, psi = self._chi_psi(k, self.a, self.b, c, d)
            Vk = 2.0 / (self.b - self.a) * (chi - psi)
        else:
            c, d = self.a, 0.0
            chi, psi = self._chi_psi(k, self.a, self.b, c, d)
            Vk = 2.0 / (self.b - self.a) * (-chi + psi)

        # COS formula
        cos_term = np.cos(u * (self.x - self.a))
        weights = np.ones(self.N)
        weights[0] *= 0.5

        return np.exp(-self.model.r * self.model.T) * np.dot(weights * Vk, phi.real * cos_term)

## Cosine methos for dicrete barrier options

class FourierCosineBarrierOption:
    def __init__(self, model, S0, K, H, N, M, T, call=True, rebate=0.0):
        self.model = model
        self.S0 = S0
        self.K = K
        self.H = H
        self.N = N
        self.M = M
        self.T = T
        self.dt = T / M
        self.call = call
        self.rebate = rebate
        self.x0 = np.log(S0 / K)
        self.h = np.log(H / K)
        self.a, self.b = model.truncation_range()

    def _chi_psi(self, k, c, d):
        a, b = self.a, self.b
        kpi = k * np.pi / (b - a)
        chi = (np.cos(kpi * (d - a)) * np.exp(d) -
               np.cos(kpi * (c - a)) * np.exp(c) +
               kpi * (np.sin(kpi * (d - a)) * np.exp(d) -
                      np.sin(kpi * (c - a)) * np.exp(c))) / (1 + kpi**2)
        psi = np.where(k == 0, d - c,
                       (np.sin(kpi * (d - a)) - np.sin(kpi * (c - a))) / kpi)
        return chi, psi

    def _payoff_coefficients(self):
        k = np.arange(self.N)
        if self.call:
            chi, psi = self._chi_psi(k, self.a, self.b)
            V = 2 / (self.b - self.a) * (chi - psi)
        else:
            chi, psi = self._chi_psi(k, self.a, self.h)
            V = 2 / (self.b - self.a) * (-chi + psi)
        return V

    def price(self):
        k = np.arange(self.N)
        u = k * np.pi / (self.b - self.a)
        x = self.x0
        phi = self.model.characteristic_function(u)

        # Initialize payoff at maturity
        V = self._payoff_coefficients()

        # Backward recursion over M monitoring times
        for _ in range(self.M):
            # COS formula update (Eq. 12 from the paper)
            F = np.real(self.model.characteristic_function(u) *
                        np.exp(-1j * u * self.a)) * V
            V = np.exp(-self.model.r * self.dt) * F

            # Barrier condition: v(x, t) = rebate if x >= h
            chi, psi = self._chi_psi(k, self.h, self.b)
            V_barrier = 2 / (self.b - self.a) * self.rebate * psi
            V = np.where(np.arange(self.N) < self.N * (self.h - self.a) / (self.b - self.a),
                         V, V_barrier)

        # Final COS price
        weights = np.ones(self.N)
        weights[0] *= 0.5
        cos_term = np.cos(u * (x - self.a))
        return np.exp(-self.model.r * self.dt) * np.dot(weights * V, cos_term)
