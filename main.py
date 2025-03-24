"""
functions.py - Core implementation of COS method for option pricing
"""

import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Analytical Black-Scholes price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def cos_transform(S, K, T, r, sigma, option_type='call', N=100, L=10):
    """COS method for European options"""
    # Characteristic function (Black-Scholes)
    def cf(u):
        return np.exp((r - 0.5*sigma**2)*1j*u*T - 0.5*sigma**2*u**2*T)
    
    # Truncation range [a,b]
    c1 = r - 0.5*sigma**2
    c2 = sigma**2
    a = c1*T - L*np.sqrt(c2*T)
    b = c1*T + L*np.sqrt(c2*T)
    
    # COS coefficients
    k = np.arange(N)
    u = k*np.pi/(b-a)
    F_k = 2/(b-a) * np.real(cf(u) * np.exp(-1j*u*a))
    
    # Payoff coefficients
    x = np.log(S/K)
    if option_type == 'call':
        chi = (np.cos(k*np.pi*(b-x)/(b-a)) * np.exp(b) - \
              (k*np.pi/(b-a))*np.sin(k*np.pi*(b-x)/(b-a)) * np.exp(b)
        psi = np.sin(k*np.pi*(b-x)/(b-a)) * np.exp(b) - np.sin(k*np.pi*(a-x)/(b-a)) * np.exp(a)
        psi[0] = 0.5*(np.exp(b) - np.exp(a))
    
    # Series summation
    V_k = np.exp(-r*T) * F_k * (chi - psi)
    return K * np.sum(V_k * np.cos(k*np.pi*(x-a)/(b-a)))

def run_analysis():
    """Generate convergence analysis results"""
    params = {'S':100, 'K':110, 'T':1, 'r':0.05, 'sigma':0.2}
    N_values = range(10, 201, 10)
    
    results = []
    for N in N_values:
        cos_price = cos_transform(N=N, **params)
        bs_price = black_scholes(**params)
        error = abs(cos_price - bs_price)
        results.append([N, cos_price, bs_price, error])
    
    return np.array(results)

if __name__ == "__main__":
    # Generate and save results
    results = run_analysis()
    np.savetxt("output/results.csv", results, 
               delimiter=",", 
               header="N,cos_price,bs_price,error", 
               fmt="%.6f")
    
    # Create convergence plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(results[:,0], results[:,3], 'b-o')
    plt.xlabel("Number of COS terms (N)", fontsize=12)
    plt.ylabel("Absolute Error", fontsize=12)
    plt.title("COS Method Convergence", fontsize=14)
    plt.grid(True)
    plt.savefig("output/figures/convergence.png", dpi=300)
