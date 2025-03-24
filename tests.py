from scr import SBM, FourierCosineMethod
import numpy as np

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
