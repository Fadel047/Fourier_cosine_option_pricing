# COS Method for Option Pricing

This project implements the Fourier-Cosine (COS) method for option pricing as introduced by Fang & Oosterlee (2008, 2009). It covers pricing of:

- European options (calls and puts)
- Discrete barrier options (up-and-out)

The project is written in Python and structured for clarity, performance, and reproducibility.

---

## 1. Theoretical Background

The COS method is a numerical method based on the Fourier-cosine series expansion of the probability density function. It relies on the availability of the characteristic function of the log-price process, which is known for many financial models such as the Black-Scholes model (GBM) and the Heston model.

For European options, the price is approximated as:

```
Price ≈ exp(-rT) * Σ [ Re( φ(u_k) * exp(-i * u_k * a) ) * V_k ]
```

where:
- `u_k = kπ / (b - a)` for `k = 0, ..., N-1`
- `φ(u)` is the characteristic function of the log-price
- `V_k` are the Fourier-cosine coefficients of the payoff function
- `[a, b]` is a truncation range that captures the probability mass of the log-price

For discrete barrier options (e.g. up-and-out), the COS method performs backward propagation in time using the characteristic function and updates the value only for log-prices below the barrier at each monitoring date.

---

## 2. European Option Pricing

Implemented in `FourierCosineMethod` using the standard Brownian motion (GBM/Black-Scholes model).

### Features
- High accuracy with a small number of terms `N`
- Very fast computation, suitable for large grids of strikes

### Results
- Exponential convergence of error with respect to `N`
- Smooth price curve across strikes

### Figures
- `figures/european_call_vs_strike.png`
- `figures/european_call_error_convergence.png`

---

## 3. Discrete Barrier Option Pricing

Implemented in `FourierCosineBarrierOption`. We price **up-and-out** barrier options with discrete monitoring dates.

### Pricing Formula


Let `h = log(H / K)` be the log-barrier level. At maturity `T`, the payoff is:

    v(x, T) = max(S_T - K, 0)     if x < h
              R_b                 otherwise

At each monitoring date `t_m`, we update the cosine coefficients using:

    V_k(t_{m-1}) = exp(-r * Δt) * Re( φ(u_k) * exp(-i * u_k * a) ) * V_k(t_m)

Then:

- For `x >= h`, the value is replaced with the rebate
- For `x < h`, we keep the continuation value

This is repeated backward in time over the `M` monitoring steps.

### Features:
- Efficient handling of M monitoring dates
- Rebate handled analytically
- Uses COS coefficients at each step

**Results:**
- Price decreases as strike increases (closer to barrier)
- Price decreases as barrier gets closer to spot

**Figures:**
- `figures/barrier_vs_strike.png`
- `figures/barrier_vs_barrier.png`

---

## 4. Project Structure

```
├── scr.py              # Core COS implementations
├── tests.py            # Test and visualization routines
├── figures/            # Output plots
└── readme.md           # Report / documentation
```

---

## 5. References

- Fang, F. & Oosterlee, C.W. (2008). A Novel Option Pricing Method Based on Fourier-Cosine Series Expansions.

---

## 6. To Do

- [ ] Add support for Heston model
- [ ] Add down-and-out barrier options
- [ ] Compare with Carr-Madan FFT method

