import numpy as np
import matplotlib.pylab as plt

import scipy.integrate as integrate
from scipy.special import gamma as gamma_func
from scipy import optimize


## Black & Scholes

class SBM(object):
    '''
    Standard Brownian Motion Object
    '''
    def __init__(self, sigma, rf, mat):
      '''
      Initialize the class.
      :param sigma: Volatility of the brownian motion
      :param rf: Risk Free rate of the economy
      :param mat: Maturity of the option
      '''
      self.sigma = sigma
      self.rf = rf
      self.mat = mat

    def characteristic_function(self, grid):
      '''
      Characteristic function of the SBM
      :grid: Integral discretization
      :return: Values of SBM's characteristic function given a grid
      '''
      phi = np.exp(grid * self.rf * self.mat * 1j
              - grid**2 * self.sigma**2 * self.mat / 2)
      return phi

    def integral_truncation(self):
      '''
      Truncation range to compute the ingration. Eq. 74 in the paper
      :param x_0: log-moneyness ln(S_0/K)
      :return: a and b, bound of integration
      '''

      L = 10
      c1 = (self.rf - 0.5 * self.sigma**2 * self.mat)
      c2 = self.sigma**2 * self.mat

      a = c1 - L * c2**.5
      b = c1 + L * c2**.5
      return a, b

## Heston model

class Heston(object):
  def __init__(self, kappa, theta, rho, eta, rf, sigma, mat):
    '''
    :param kappa: Rate at which the dynamic return to its long term value
    :param theta: Long variance, or long-run average variance of the price
    :param rho: Correlation of the two Wienner processes
    :param eta: Volatility of volatility
    :param sigma: Initial variance of the asset price
    :param rf: Risk free rate
    :param mat: Maturity
    '''
    self.kappa = kappa
    self.theta = theta
    self.rho = rho
    self.eta = eta
    self.sigma = sigma
    self.rf = rf
    self.mat = mat


  def characteristic_function(self, grid):
    '''
    Characteristic function of the Heston model
    :grid: Integral discretization
    :return: Values of Heston's characteristic function given a grid
    '''
    # Keep the parameters into variables in order to simplify the notation
    kappa, theta, sigma  = self.kappa, self.theta, self.sigma,
    rho, eta, rf, mat = self.rho, self.eta, self.rf, self.mat

    # Compute D and G
    D = np.sqrt( (kappa - 1j*rho*eta*grid)**2 + (grid**2 + 1j*grid)*eta**2 )
    G = (kappa - 1j*rho*eta*grid - D) / (kappa - 1j*rho*eta*grid + D)

    # Compute the two part of phi
    first_part = np.exp( 1j*grid*rf*mat + sigma/eta**2 \
                        * ((1-np.exp(-D*mat)) / (1-G*np.exp(-D*mat))) \
                        * (kappa - 1j*rho*eta*grid - D) )

    second_part = np.exp( kappa*theta/eta**2 \
                         * (mat * (kappa - 1j*rho*eta*grid - D) - 2*np.log((1-G*np.exp(-D*mat)) / (1-G))) )

    # Construct the characteristic function of the Heston model
    phi = first_part * second_part

    return phi

  def integral_truncation(self):
    '''
    Truncation range to compute the ingration. Eq. 74 in the paper
    :param x_0: log-moneyness ln(S_0/K)
    :return: a and b, bound of integration
    '''

    # Keep the parameters into variables in order to simplify the notation
    kappa, theta, sigma  = self.kappa, self.theta, self.sigma
    rho, eta, rf, mat = self.rho, self.eta, self.rf, self.mat

    L = 12
    c1 = rf * mat \
        + (1 - np.exp(-kappa * mat)) \
        * (theta - sigma)/2 / kappa - theta * mat / 2

    c2 = 1/(8 * kappa**3) \
        * (eta * mat * kappa * np.exp(-kappa * mat) \
        * (sigma - theta) * (8 * kappa * rho - 4 * eta) \
        + kappa * rho * eta * (1 - np.exp(-kappa * mat)) \
        * (16 * theta - 8 * sigma) + 2 * theta * kappa * mat \
        * (-4 * kappa * rho * eta + eta**2 + 4 * kappa**2) \
        + eta**2 * ((theta - 2 * sigma) * np.exp(-2*kappa*mat) \
        + theta * (6 * np.exp(-kappa*mat) - 7) + 2 * sigma) \
        + 8 * kappa**2 * (sigma - theta) * (1 - np.exp(-kappa*mat)))

    a = c1 - L * np.abs(c2)**.5
    b = c1 + L * np.abs(c2)**.5
    return a, b

## Cosine Method

class Fourier_cosine_method(object):
  def __init__(self, model, log_moneyness, call_option, grid_size):
    '''
    :param model: Model to be used to price the option
    :param log_moneyness: Float or list of log-moneyness given a strike K and an underlying price S_0
    :param call_option: Boolean to specify wether it is a call or a put option to price
    :param grid_size: Number of point to be used on the grid
    '''
    self._model = model
    self._log_moneyness = log_moneyness
    self._grid_size = grid_size
    self._call_option = call_option

    # Get the bound of integration
    self._a , self._b = self._model.integral_truncation()

  def cosine_expansion(self):
    '''
    This function compute the price of an option given a certain model using the Fourier cosine expansion methods
    :return: Option premium
    '''

    # Compute the k
    k = np.arange(self._grid_size, dtype=complex)[:, np.newaxis]

    # Compute V for the call and put option using the xhi and psi cosine series coefficients
    if self._call_option:
      xhi, psi = self.get_xhi_psi_coefficients(k, 0, self._b)
      v_mat = 2 / (self._b - self._a) * (xhi - psi)
    else:
      xhi, psi = self.get_xhi_psi_coefficients(k, self._a, 0)
      v_mat = 2 / (self._b - self._a) * (-xhi + psi)

    # Compute the model characteristic function
    char_mat = self._model.characteristic_function(k * np.pi/(self._b-self._a))

    # Exponential part
    exp_mat = np.exp(1j * k * np.pi * (self._a - self._log_moneyness)/(self._b - self._a))

    # Grid of weight with 1/2 to start
    weights = np.append(.5, np.ones(self._grid_size-1))

    return np.exp(-self._model.rf * self._model.mat) * np.dot(weights, char_mat * exp_mat * v_mat).real

  def get_xhi_psi_coefficients(self, k, x_1, x_2):
    '''
    This function compute the xhi and psi function for given parameters.
    :param k: Log-moneyness of the options
    :param x_1: Lower bound of the inner integration interval
    :param x_2: Upper bound of the inner integration interval
    :return: The cosine series coefficients xhi and psi
    '''

    # Xhi cosine serie coefficients
    xhi = (np.cos(k * np.pi * (x_2-self._a)/(self._b-self._a)) * np.exp(x_2) - np.cos(k * np.pi * (x_1-self._a)/(self._b-self._a)) * np.exp(x_1)
          + k * np.pi/(self._b-self._a) * (np.sin(k * np.pi * (x_2-self._a)/(self._b-self._a)) * np.exp(x_2) - np.sin(k * np.pi * (x_1-self._a)/(self._b-self._a)) * np.exp(x_1)))\
          / (1 + (k * np.pi/(self._b-self._a))**2)

    # Psi cosine serie coefficients
    psi = (np.sin(k[1:] * np.pi * (x_2-self._a)/(self._b-self._a)) - np.sin(k[1:] * np.pi * (x_1-self._a)/(self._b-self._a))) / (k[1:] * np.pi/(self._b-self._a))
    psi = np.vstack([(x_2 - x_1) * np.ones_like(self._a), psi])
    return xhi, psi

## Function to compute option price using Fourier Carr-Madan method

class Fourier_Carr_Madan_Method(object):

  def __init__(self):
    pass

  @staticmethod
  def Call_Price_Heston(S, K, T, r, kappa, theta, nu, rho, V_0, alpha=1, L=1000):
    '''
    :param S: Initial Price
    :param K: Strike
    :param r: risk free rate
    :param kappa: Rate at which the dynamic return to its long term value
    :param theta: Long variance, or long-run average variance of the price
    :param nu: Long term value of the dynamic
    :param rho: correlation of the two Wienner processes
    :param V_0: initial volatility
    :param alpha: Damping factor (alpha>0) typically alpha = 1
    :param L: Truncation bound for the integral
    '''
    # Complex number
    i = complex(0,1)

    b = lambda x: (kappa - 1j*rho*nu*x)
    gamma = lambda x: (np.sqrt(nu**(2) * (x**2+1j+x) + b(x)**2))
    a = lambda x: (b(x) / gamma(x)) * np.sinh(T*.5*gamma(x))
    c = lambda x:(gamma(x) * np.cosh(.5*T*gamma(x))) / np.sinh(T*.5*gamma(x)+b(x))
    d = lambda x:(kappa*theta*T*b(x) / nu**2)

    f = lambda x:(1j * (np.log(S)+r*T) * x + d(x))
    g = lambda x:(np.cosh(T*.5*gamma(x)) + a(x)) ** (2*kappa*theta / nu**2)
    h = lambda x:(- (x**2+1j*x) * V_0 / c(x))

    phi = lambda x:(np.exp(f(x)) * np.exp(h(x)/g(x))) # Characteristics function

    integrand = lambda x:( np.real( phi(x-1j*(alpha+1)) / ((alpha+1j*x) * (alpha+1+1j*x)) ) * np.exp(-1j*np.log(K)*x) )
    integral = integrate.quad(integrand, 0, L)
    price = ( np.exp(-r*T - alpha*np.log(K)) / np.pi ) * integral[0]
    return price

## Examples

# SBM Cosine Expansion price
price, strike = 100, 90
riskfree, maturity = 0, 180/365
sigma = .15
log_moneyness = np.log(price/strike)

model = SBM(sigma, riskfree, maturity)
method = Fourier_cosine_method(model, log_moneyness, True, 2**10)
premium = method.cosine_expansion()
print(premium)

# Heston Cosine Expansion price
kappa = 1.5768
theta = .12**2
eta = .5751
rho = -.0
sigma = .12**2
model = Heston(kappa, theta, rho, eta, riskfree, sigma, maturity)
method = Fourier_cosine_method(model, log_moneyness, True, 2**10)
premium = method.cosine_expansion()
print(premium)

## SBM Cosine Expansion price

# Number of option to price
grid_size = 2000 # nb of options to price

# Initialize parameters
riskfree, maturity = 0, 1/12
sigma = .15
price = 1
strike = np.exp(np.linspace(-.1, .1, grid_size))
log_moneyness = np.log(price/strike)
call_option = False

# Initialize the model and method
model = SBM(sigma, riskfree, maturity)
method = Fourier_cosine_method(model, log_moneyness, call_option, grid_size)

# Compute premium of options
premium = method.cosine_expansion()

# Plot the strike/premium
plt.plot(strike, premium)
plt.xlabel("Moneyness S/K")
plt.ylabel("Premium")
if call_option:
  plt.title("GBM Log Moneyness Call Option")
  plt.savefig('GMB_Call.png')
else:
  plt.title("GBM Log Moneyness Put Option")
  plt.savefig('GMB_Put.png')

plt.show()
