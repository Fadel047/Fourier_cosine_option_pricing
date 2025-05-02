import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

class COSMethod:
    
    def __init__(self, S0=100, r=0.05, T=1, sigma=0.2, model="GBM", **model_params):

        """
        Initialize the model parameters.

        Args:
            S0 (float): Initial asset price.
            r (float): Risk-free interest rate.
            T (float): Option maturity (in years).
            sigma (float): Volatility (for Black-Scholes).
            model (str): Underlying model ("GBM", "Heston", "CGMY").
            model_params (dict): Model-specific parameters (e.g., lambda, eta for Heston).
            ...
        """
        self.S0 = S0
        self.r = r
        self.T = T
        self.sigma = sigma
        self.model = model
        self.model_params = model_params
        
        # Default parameters for the models
        if model == "Heston":
            self.lambda_ = model_params.get("lambda", 1.5768)
            self.eta = model_params.get("eta", 0.5751)
            self.u0 = model_params.get("u0", 0.0175)
            self.u_bar = model_params.get("u_bar", 0.0398)
            self.rho = model_params.get("rho", -0.5711)
        elif model == "CGMY":
            self.C = model_params.get("C", 1.0)
            self.G = model_params.get("G", 5.0)
            self.M = model_params.get("M", 5.0)
            self.Y = model_params.get("Y", 0.5)
    
    def characteristic_function(self, omega):

        """
        Compute the characteristic function based on the selected model.
    
        Args:
        omega (float or array): Fourier frequency/frequencies.
        
        Returns:
        complex or array: Value(s) of the characteristic function.
        """
        
        if self.model == "GBM":
            return np.exp(1j * omega * (self.r - 0.5 * self.sigma**2) * self.T 
                          - 0.5 * omega**2 * self.sigma**2 * self.T)
        
        elif self.model == "Heston":
            # Characteristic function for Heston
            lambda_ = self.lambda_
            eta = self.eta
            u0 = self.u0
            u_bar = self.u_bar
            rho = self.rho
            
            D = np.sqrt((lambda_ - 1j * rho * eta * omega)**2 + (omega**2 + 1j * omega) * eta**2)
            G = (lambda_ - 1j * rho * eta * omega - D) / (lambda_ - 1j * rho * eta * omega + D)
            exp_DT = np.exp(-D * self.T)
            
            return np.exp(
                1j * omega * (self.r - 0.5 * u0) * self.T 
                + u0 / eta**2 * (1 - exp_DT) / (1 - G * exp_DT) * (lambda_ - 1j * rho * eta * omega - D)
                + (lambda_ * u_bar / eta**2) * (
                    (lambda_ - 1j * rho * eta * omega - D) * self.T 
                    - 2 * np.log((1 - G * exp_DT) / (1 - G))
            )
        
        elif self.model == "CGMY":
            # Characteristic function for CGMY
            C, G, M, Y = self.C, self.G, self.M, self.Y
            return np.exp(
                1j * omega * (self.r - 0.5 * self.sigma**2) * self.T 
                - 0.5 * omega**2 * self.sigma**2 * self.T
                + C * self.T * np.math.gamma(-Y) * (
                    (M - 1j * omega)**Y - M**Y 
                    + (G + 1j * omega)**Y - G**Y
                )
            )
    
    def compute_integration_range(self, K, L=10):
        """
        Compute the truncation interval [a, b] based on the cumulants.
        
        Args:
            K (float): Strike de l'option.
            L (float): Paramètre de contrôle (défaut: 10).
            
        Returns:
            tuple: (a, b)
        """
        x0 = np.log(self.S0 / K)
        
        if self.model == "GBM":
            c1 = (self.r - 0.5 * self.sigma**2) * self.T
            c2 = self.sigma**2 * self.T
            c4 = 0.0
        elif self.model == "Heston":
            # Cumulants for Heston (see the reference)
            c1 = (self.r - 0.5 * self.u0) * self.T
            c2 = self.u0 * self.T  # Simplified approximation
            c4 = 0.0  # Simplification
        else:
            raise NotImplementedError("Cumulants not implemented for this model.")
        
        a = x0 + c1 - L * np.sqrt(c2 + np.sqrt(c4))
        b = x0 + c1 + L * np.sqrt(c2 + np.sqrt(c4))
        return a, b
    
    def compute_cosine_coefficients(self, K, option_type="call", N=256, a=None, b=None):
        """
        Compute the Vₖ coefficients for the option payoff.
        
        Args:
            K (float): Strike.
            option_type (str): "call" ou "put".
            N (int): Nombre de termes dans l'expansion COS.
            a, b (float): Bornes de l'intervalle de troncature.
            
        Returns:
            array: Coefficients V_k de taille N.
        """
        if a is None or b is None:
            a, b = self.compute_integration_range(K)
        
        k = np.arange(N)
        omega_k = k * np.pi / (b - a)
        
        # Implement V_k for each option type
        if option_type == "call":
            c, d = 0.0, b
            chi_k = (np.cos(omega_k * (d - a)) * np.exp(d) 
                     - np.cos(omega_k * (c - a)) * np.exp(c)
                     + omega_k * np.sin(omega_k * (d - a)) * np.exp(d)
                     - omega_k * np.sin(omega_k * (c - a)) * np.exp(c)) / (1 + omega_k**2)
            psi_k = (np.sin(omega_k * (d - a)) - np.sin(omega_k * (c - a))) / omega_k
            psi_k[0] = d - c  # Cas k=0
            V_k = (2 / (b - a)) * K * (chi_k - psi_k)
        
        elif option_type == "put":
            c, d = a, 0.0
            chi_k = (np.cos(omega_k * (d - a)) * np.exp(d) 
                     - np.cos(omega_k * (c - a)) * np.exp(c)
            psi_k = (np.sin(omega_k * (d - a)) - np.sin(omega_k * (c - a))) / omega_k
            psi_k[0] = d - c
            V_k = (2 / (b - a)) * K * (-chi_k + psi_k)
        
        return V_k
    
    def price_european_option(self, K, option_type="call", N=256):
        """
        Price a European option using the COS method.
        
        Args:
            K (float): Strike.
            option_type (str): "call" ou "put".
            N (int): Nombre de termes COS.
            
        Returns:
            float: Prix de l'option.
        """
        a, b = self.compute_integration_range(K)
        V_k = self.compute_cosine_coefficients(K, option_type, N, a, b)
        
        k = np.arange(N)
        omega_k = k * np.pi / (b - a)
        phi_k = self.characteristic_function(omega_k)
        
        x0 = np.log(self.S0 / K)
        term = np.exp(-1j * omega_k * a) * phi_k * V_k
        price = np.exp(-self.r * self.T) * np.real(np.sum(term)) * (0.5 if N > 0 else 1.0)
        
        return price
    
    def price_discrete_barrier_option(self, K, H, option_type="call", M=10, N=256):

        """
        Price a discrete barrier option (up-and-out) using the COS method.
        
        Args:
            K (float): Strike price.
            H (float): Barrier level.
            option_type (str): "call" or "put".
            M (int): Number of monitoring dates.
            N (int): Number of COS expansion terms.
        
        Returns:
            float: Option price.
        """
        a, b = self.compute_integration_range(K)
        h = np.log(H / K)
        
        # At the maturity
        V_k = np.zeros(N)
        for k in range(N):
            if option_type == "call":
                if H > K:  # Up-and-out call
                    V_k[k] = (2 / (b - a)) * K * (
                        self._compute_chi_k(k, 0, h, a, b) 
                        - self._compute_psi_k(k, 0, h, a, b)
                    )
            else:
                raise NotImplementedError("Autres types de barrières non implémentés.")
        
        # Backward recursion
        delta_t = self.T / M
        for m in range(M - 1, 0, -1):
            phi_k = self.characteristic_function(k * np.pi / (b - a))
            term = np.exp(-1j * k * np.pi * a / (b - a)) * phi_k * V_k
            C_k = np.exp(-self.r * delta_t) * np.real(np.sum(term))
            
            # Update V_k for t_{m-1}
            for k in range(N):
                if option_type == "call":
                    V_k[k] = C_k + (2 / (b - a)) * K * (
                        self._compute_psi_k(k, h, b, a, b)  # Rebate if H is touched
                    )
        
        # final price (at t=0)
        price = np.exp(-self.r * delta_t) * np.real(np.sum(
            np.exp(-1j * k * np.pi * a / (b - a)) 
            * self.characteristic_function(k * np.pi / (b - a)) 
            * V_k
        ))
        
        return price
    
    def _compute_chi_k(self, k, c, d, a, b):
        """Calcule chi_k pour [c, d] (voir eq. 22)."""
        term = (np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) 
               - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
        term += (k * np.pi / (b - a)) * (
            np.sin(k * np.pi * (d - a) / (b - a)) * np.exp(d)
            - np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
        )
        return term / (1 + (k * np.pi / (b - a))**2)
    
    def _compute_psi_k(self, k, c, d, a, b):
        """Calcule psi_k pour [c, d] (voir eq. 23)."""
        if k == 0:
            return d - c
        return (np.sin(k * np.pi * (d - a) / (b - a)) 
                - np.sin(k * np.pi * (c - a) / (b - a))) * (b - a) / (k * np.pi)


# Example
if __name__ == "__main__":
    # settings
    S0 = 100
    K = 100
    r = 0.05
    T = 1
    sigma = 0.2
    
    # 1. Pricing of european option (Call)
    cos = COSMethod(S0=S0, r=r, T=T, sigma=sigma, model="GBM")
    european_price = cos.price_european_option(K, option_type="call", N=128)
    print(f"Price of the European option (Call): {european_price:.4f}")
    
    # 2. Pricing of discrete barrier option (Up-and-Out Call)
    H = 120  # Barrier level
    barrier_price = cos.price_discrete_barrier_option(K, H, option_type="call", M=10, N=128)
    print(f"Price of the barrier option (Up-and-Out Call): {barrier_price:.4f}")
