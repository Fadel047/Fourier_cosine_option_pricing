"""Run a European call pricing example."""

from fourier_cosine_option_pricing import BlackScholesModel, FourierCosineMethod


def main():
    model = BlackScholesModel(sigma=0.2, rate=0.05, maturity=1.0)
    pricer = FourierCosineMethod(model, spot=100.0, strike=100.0, grid_size=256, call=True)
    print(f"European call price: {pricer.price():.6f}")


if __name__ == "__main__":
    main()
