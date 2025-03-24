# COS Method for Option Pricing

This project implements the Fourier-Cosine (COS) method for option pricing as introduced by Fang & Oosterlee (2008, 2009). It covers pricing of:

- European options (calls and puts)
- Discrete barrier options (up-and-out)

The project is written in Python and structured for clarity, performance, and reproducibility.

---

## 1. Theoretical Background

The COS method is a spectral method based on Fourier-cosine series expansion of the probability density function. It leverages the availability of the characteristic function of the underlying process (e.g., Brownian motion, Heston model).

For European options, the price is approximated by:

\[ c(x) \approx e^{-rT} \sum_{k=0}^{N-1} Re\left[ \phi\left(\frac{k\pi}{b - a}\right) e^{-i k \pi \frac{a}{b - a}} \right] V_k \]

Where `V_k` are Fourier-cosine coefficients of the payoff function, and `\phi` is the characteristic function.

For discretely monitored barrier options (e.g., up-and-out):
- A similar backward recursion is used as in Bermudan options
- At each monitoring date, the option value is set to the rebate value if the log-price crosses the barrier

---

## 2. European Option Pricing

Implemented in `FourierCosineMethod` using the SBM model (lognormal / Black-Scholes dynamics).

**Features:**
- High accuracy with small N
- Fast pricing of large strike grids

**Results:**
- Exponential convergence in number of terms N
- Price vs Strike is smooth and matches Black-Scholes values

**Figures:**
- `figures/european_call_vs_strike.png`
- `figures/european_call_error_convergence.png`

---

## 3. Discrete Barrier Option Pricing

Implemented in `FourierCosineBarrierOption`. We price up-and-out options using backward propagation.

**Features:**
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
- Fang, F. & Oosterlee, C.W. (2009). Pricing Early-Exercise and Discrete Barrier Options by Fourier-Cosine Expansions.

---

## 6. To Do

- [ ] Add support for Heston model
- [ ] Add Bermudan option pricing with early exercise optimization
- [ ] Add comparison to Carr-Madan FFT pricing

