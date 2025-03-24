from scr import SBM, FourierCosineMethod
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

def test_european_option():
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    N = 256

    model = SBM(sigma, r, T)
    pricer = FourierCosineMethod(model, S0, K, N, call=True)
    price = pricer.price()
    print(f"European Call Price (COS): {price:.6f}")

if __name__ == "__main__":
    test_european_option()

#Plot Price vs Strike (European options)

def plot_price_vs_strike():
    S0 = 100
    T = 1
    r = 0.05
    sigma = 0.2
    N = 256
    strikes = np.linspace(80, 120, 100)
    prices = []

    model = SBM(sigma, r, T)
    for K in strikes:
        pricer = FourierCosineMethod(model, S0, K, N, call=True)
        prices.append(pricer.price())

    plt.figure()
    plt.plot(strikes, prices, label="COS Price")
    plt.xlabel("Strike")
    plt.ylabel("Call Option Price")
    plt.title("European Call Option - COS Method")
    plt.grid(True)
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/european_call_vs_strike.png")
    plt.close()

#Error Convergence vs N

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def plot_error_convergence():
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    true_price = black_scholes_call(S0, K, T, r, sigma)
    Ns = [2**i for i in range(3, 11)]
    errors = []

    model = SBM(sigma, r, T)
    for N in Ns:
        pricer = FourierCosineMethod(model, S0, K, N, call=True)
        price = pricer.price()
        errors.append(abs(price - true_price))

    plt.figure()
    plt.plot(Ns, errors, marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel("Number of Terms (N)")
    plt.ylabel("Absolute Error")
    plt.title("Error Convergence of COS Method")
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/european_call_error_convergence.png")
    plt.close()
