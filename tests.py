"""Legacy entrypoint; automated tests are in ``tests/``."""

from fourier_cosine_option_pricing import BlackScholesModel, FourierCosineMethod


def demo_price():
    model = BlackScholesModel(sigma=0.2, rate=0.05, maturity=1.0)
    return FourierCosineMethod(model, spot=100.0, strike=100.0, grid_size=256, call=True).price()


if __name__ == "__main__":
    print(f"European call price: {demo_price():.6f}")
