import numpy as np
from scipy.stats import norm

from fourier_cosine_option_pricing import BlackScholesModel, FourierCosineMethod, HestonModel


def black_scholes_call(spot, strike, maturity, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)


def test_cos_matches_black_scholes_call():
    model = BlackScholesModel(sigma=0.2, rate=0.05, maturity=1.0)
    price = FourierCosineMethod(model, 100.0, 100.0, grid_size=256, call=True).price()
    reference = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2)
    assert abs(price - reference) < 1e-6


def test_cos_matches_black_scholes_put():
    model = BlackScholesModel(sigma=0.2, rate=0.05, maturity=1.0)
    price = FourierCosineMethod(model, 100.0, 100.0, grid_size=256, call=False).price()
    call = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2)
    reference = call - 100.0 + 100.0 * np.exp(-0.05)
    assert abs(price - reference) < 1e-6


def test_heston_characteristic_function_is_finite_and_normalized():
    model = HestonModel(kappa=1.5, theta=0.04, rho=-0.7, eta=0.3, rf=0.02, sigma=0.04, mat=1.0)
    values = model.characteristic_function(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(values))
    assert np.isclose(values[0], 1.0)
