"""Fourier-Cosine option pricing package."""

from .methods.cosine import FourierCosineMethod
from .methods.carr_madan import FourierCarrMadanMethod
from .models.black_scholes import BlackScholesModel
from .models.heston import HestonModel

# Compatibility with the original project naming.
SBM = BlackScholesModel

__all__ = ["SBM", "BlackScholesModel", "HestonModel", "FourierCosineMethod", "FourierCarrMadanMethod"]
