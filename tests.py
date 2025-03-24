"""
tests.py - Unit tests for option pricing functions
"""

import numpy as np
from functions import black_scholes, cos_transform

def test_black_scholes():
    """Verify Black-Scholes implementation"""
    price = black_scholes(100, 110, 1.0, 0.05, 0.2)
    assert np.isclose(price, 7.9656, atol=0.01), "Black-Scholes formula error"

def test_cos_convergence():
    """Verify COS method convergence"""
    prices = [cos_transform(100, 110, 1.0, 0.05, 0.2, N=N) for N in [50, 100, 200]]
    deviations = [abs(p - 7.9656) for p in prices]
    assert all(d < 0.1 for d in deviations), "COS method not converging"
    assert deviations[-1] < deviations[0], "Error not decreasing with N"

def test_put_parity():
    """Verify put-call parity"""
    call = cos_transform(100, 110, 1.0, 0.05, 0.2, option_type='call')
    put = cos_transform(100, 110, 1.0, 0.05, 0.2, option_type='put')
    assert np.isclose(call - put, 100 - 110*np.exp(-0.05), atol=0.1), "Put-call parity violation"

if __name__ == "__main__":
    test_black_scholes()
    test_cos_convergence()
    test_put_parity()
    print("All tests passed successfully!")
