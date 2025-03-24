"""
functions.py - Implémentation de la méthode COS pour le pricing d'options
"""

import numpy as np
from scipy.stats import norm

def bs_analytical(S, K, T, r, sigma, option_type='call'):
    """Prix analytique Black-Scholes"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def cos_pricing(S, K, T, r, sigma, option_type='call', N=100, L=10):
    """Méthode COS pour options européennes"""
    # Fonction caractéristique Black-Scholes
    def cf(u):
        return np.exp((r - 0.5*sigma**2)*1j*u*T - 0.5*sigma**2*u**2*T)
    
    # Intervalle de troncature [a,b]
    c1 = r - 0.5*sigma**2
    c2 = sigma**2
    a = c1*T - L*np.sqrt(c2*T)
    b = c1*T + L*np.sqrt(c2*T)
    
    # Coefficients COS
    k = np.arange(N)
    u = k*np.pi/(b-a)
    F_k = 2/(b-a) * np.real(cf(u) * np.exp(-1j*u*a))
    
    # Coefficients de payoff
    x = np.log(S/K)
    if option_type == 'call':
        chi = (np.cos(k*np.pi*(b-x)/(b-a)) * np.exp(b) - 
              (k*np.pi/(b-a))*np.sin(k*np.pi*(b-x)/(b-a)) * np.exp(b))
        psi = np.sin(k*np.pi*(b-x)/(b-a)) * np.exp(b) - np.sin(k*np.pi*(a-x)/(b-a)) * np.exp(a)
        psi[0] = 0.5*(np.exp(b) - np.exp(a))
    
    # Somme discrétisée
    V_k = np.exp(-r*T) * F_k * (chi - psi)
    price = K * np.sum(V_k * np.cos(k*np.pi*(x-a)/(b-a)))
    
    return price

def generate_results():
    """Génère les résultats pour le rapport"""
    params = {'S':100, 'K':110, 'T':1, 'r':0.05, 'sigma':0.2}
    N_values = range(10, 201, 10)
    
    results = []
    for N in N_values:
        cos_price = cos_pricing(N=N, **params)
        bs_price = bs_analytical(**params)
        error = abs(cos_price - bs_price)
        results.append([N, cos_price, bs_price, error])
    
    return np.array(results)
