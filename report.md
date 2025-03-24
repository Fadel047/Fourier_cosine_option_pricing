# COS Method for Option Pricing - Analysis Report

## 1. Methodology
Implemented the Fourier-COS method for European options with:
- Black-Scholes characteristic function
- Dynamic truncation range [a, b] = [c₁T ± L√(c₂T)]
- Cosine series expansion with N terms

## 2. Results

### Convergence Analysis
![Convergence Plot](output/figures/convergence.png)

Key observations:
- Exponential convergence for N > 50
- Machine precision reached at N ≈ 160
- Relative error < 0.1% for N ≥ 100

## 3. Benchmark
| N  | COS Price | BS Price | Error |
|----|----------|---------|-------|
| 20 | 7.9521   | 7.9656  | 0.0135|
| 100| 7.9655   | 7.9656  | 0.0001|
