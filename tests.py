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

## Test dicrete barrier options

def test_discrete_barrier_option():
    S0 = 100
    K = 100
    H = 120  # up-and-out
    T = 1
    r = 0.05
    sigma = 0.2
    N = 256
    M = 12
    rebate = 0.0

    model = SBM(sigma, r, T)
    pricer = FourierCosineBarrierOption(model, S0, K, H, N, M, T, call=True, rebate=rebate)
    price = pricer.price()
    print(f"Discrete Barrier Call (up-and-out) Price: {price:.6f}")


def plot_barrier_option_vs_strike():
    S0 = 100
    T = 1
    r = 0.05
    sigma = 0.2
    N = 256
    M = 12
    rebate = 0.0
    H = 120  # Fixed up-and-out barrier
    strikes = np.linspace(70, 110, 100)
    prices = []

    model = SBM(sigma, r, T)
    for K in strikes:
        pricer = FourierCosineBarrierOption(model, S0, K, H, N, M, T, call=True, rebate=rebate)
        prices.append(pricer.price())

    plt.figure()
    plt.plot(strikes, prices, label="Up-and-Out Barrier Call")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title(f"Discrete Barrier Option vs Strike (H = {H})")
    plt.grid(True)
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/barrier_vs_strike.png")
    plt.close()


#Price vs Barrier Level (for fixed strike)
def plot_barrier_option_vs_barrier():
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    N = 256
    M = 12
    rebate = 0.0
    barriers = np.linspace(105, 140, 100)  # Only up-and-out (H > S0)
    prices = []

    model = SBM(sigma, r, T)
    for H in barriers:
        pricer = FourierCosineBarrierOption(model, S0, K, H, N, M, T, call=True, rebate=rebate)
        prices.append(pricer.price())

    plt.figure()
    plt.plot(barriers, prices, label="Up-and-Out Barrier Call")
    plt.xlabel("Barrier Level (H)")
    plt.ylabel("Option Price")
    plt.title(f"Discrete Barrier Option vs Barrier (K = {K})")
    plt.grid(True)
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/barrier_vs_barrier.png")
    plt.close()

if __name__ == "__main__":
    test_european_option()
    plot_price_vs_strike()
    plot_error_convergence()
    test_discrete_barrier_option()
    plot_barrier_option_vs_strike()
    plot_barrier_option_vs_barrier()
