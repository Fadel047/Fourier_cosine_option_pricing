# COS Method for Option Pricing

This project implements the Fourier-Cosine (COS) method for pricing European and discrete barrier options in Python, based on the work of Fang & Oosterlee (2008, 2009).

---

## 🧠 What is the COS Method?

The COS method is a fast and accurate technique for pricing options when the characteristic function of the log-price is known (e.g., Black-Scholes, Heston, Lévy models). It expands the density function using a Fourier-cosine series and computes the option price via analytical integration.

### ✅ Key Features

- 📈 **Exponential convergence** for smooth densities
- ⚡ **High efficiency** (linear complexity)
- 🔁 Works with **European options** and **discrete barrier options**
- 🧩 Easy to extend to complex payoffs

---

## 📊 Implemented Features

### European Options

- ✔️ Call and Put options
- ✔️ Supports GBM (Black-Scholes) and Heston models
- ✔️ Efficient pricing for multiple strikes
- ✔️ Error convergence analysis

<div align="center">
  <img src="figures/european_call_vs_strike.png" width="400">
  <img src="figures/european_call_error_convergence.png" width="400">
</div>

---

### Discrete Barrier Options

- ✔️ Up-and-out calls (with or without rebate)
- ✔️ Handles multiple monitoring dates
- ✔️ Backward propagation of barrier condition

---

## References

- Fang, F. & Oosterlee, C.W. (2008). A Novel Option Pricing Method Based on Fourier-Cosine Series Expansions.

---

