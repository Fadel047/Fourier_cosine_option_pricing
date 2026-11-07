"""Compare COS prices with the Black-Scholes closed-form benchmark."""

import numpy as np
from scipy.stats import norm

from fourier_cosine_option_pricing import BlackScholesModel, FourierCosineMethod


def black_scholes_call(spot, strike, maturity, rate, sigma):
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)


def main():
    spot = strike = 100.0
    maturity, rate, sigma = 1.0, 0.05, 0.2
    reference = black_scholes_call(spot, strike, maturity, rate, sigma)
    model = BlackScholesModel(sigma=sigma, rate=rate, maturity=maturity)
    for grid_size in (8, 16, 32, 64, 128, 256, 512):
        price = FourierCosineMethod(model, spot, strike, grid_size, call=True).price()
        print(f"N={grid_size:>3}: COS={price:.8f}, error={abs(price-reference):.3e}")


if __name__ == "__main__":
    main()
