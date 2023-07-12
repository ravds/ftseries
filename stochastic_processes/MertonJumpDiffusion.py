import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from typing import List, Optional, Dict, Union
import warnings

# Authors: Robbie von der Schmidt, ravds@alumni.princetno.edu, Daksh Sharma

# TODO: Make sure all functionalities are well encapsulated and in keeping with design philosophies 
#       figure out better solution for magic numbers (# of poisson jumps per unit time, model parameter initialization)
#       Decide which methods should accept model parameters as inputs 

# Observations: Nelder-Mead and L-BFGS-B converging to same parameter vector

class MertonJumpDiffusion:
    """
    Create a Merton Jump Diffusion process based on Merton's 1976 paper, 'Option pricing when underlying stock returns are discontinuous',
    describing a stochastic process with geometric brownian motion and log-normally distributed IID discontinous jumps with intervals governed by a homogenous poisson process. 
    Times until next events in homogoenous poisson processes occur according to an exponential distribution, which is a special memorylessness case of the weibull distribution. 

    Parameters:
    path: np.array, default = None
    A single historical path of the time series variable. When modeling future prices, this is simply an array of historical prices, which will be transformed into an array 
    of log % increases before being passed to a fit method. 

    mu: float, default = 0
    Diffusive Drift 

    sigma: float, default = 1.0
    Diffusive Volatility

    lambda_: float, default = 1
    Average number of jumps per unit time

    mu_j: defalt, default = 0
    Mean log-jump size

    sigma_j: float, default = 1
    Jump volatility 

    optimize_method: string, default = None
    Optimization method used 
    """

    def __init__(self, path: np.array = None, mu: float = 0, sigma: float = 1, lambda_: float = 1,
                 mu_j: float = 0, sigma_j: float = 1, optimize_method: str = None):

        self.path = path
        self.mu = mu
        self.sigma = sigma
        self.lambda_ = lambda_
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.optimize_method = optimize_method
        self.likelihood_values = []

    def calculate_log_returns(self, path: np.array) -> np.array:
        """
        Calculate log % returns of each step in a path

        Parameters:
        path: np.array representing a single path

        Returns:
        np.array of log returns
        """
        return np.log(path[1:]) - np.log(path[:-1])

    def _callback(self, current_params):
        """
        A callback function for -log(likelihoods) to help us analyze optimization method performance


        Parameters:
        current_params: The current parameter values in the optimization.
        """
        self.likelihood_values.append({
            'step': len(self.likelihood_values),
            'log_likelihood': self.negative_log_likelihood(current_params),
            'mu': current_params[0],
            'sigma': current_params[1],
            'lambda': current_params[2],
            'mu_j': current_params[3],
            'sigma_j': current_params[4]
        })

    def negative_log_likelihood(self, params):
        """
        Function to calculate the total negative log-likelihood of a sequence of a log % changes 
        Note: The log funciton is monotonic over its support, (0, inf), with the advantage that a log of a product is a sum of the logarithms of the product terms 
        We will add a small epsilon to prevent passing any 0's to the logarithm, which is a fairly commmon appraoch (see most Naive Bayes implementations)

        Parameters:
        params: List of parameters [mu, sigma, lambda_, mu_j, sigma_j].

        Returns:
        Total negative log-likelihood.
        """

        # Unpack parameters
        mu, sigma, lambda_, mu_j, sigma_j = params

        # Check lambda is non-negative
        if lambda_ < 0:
            raise ValueError("Lambda must be non-negative.")

        # Compute expected jump size (in keeping with Merton's formulation)
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # Compute the adjusted drift
        mu_adj = mu - lambda_ * k

        # Compute the terms for the likelihood function. 100 is a magic number based on empirical observations that jump frequencies are small with daily time steps, 
        # so 100 jumps over any period would be extremely unlikely. That being said, a threshold should be implemented here instead 
        num_terms = 100
        ks = np.arange(num_terms)

        dt = self.dt  # use saved dt
        steps = self.calculate_log_returns(self.path)  # use saved log returns

        poisson_term = poisson.pmf(ks, lambda_ * dt)
        normal_term = norm.pdf(steps[:, None], loc=(mu_adj - sigma**2 / 2) * dt + mu_j * ks, scale=np.sqrt(sigma**2 * dt + sigma_j**2 * ks))

        likelihood_terms = poisson_term * normal_term

        # Add a small constant (epsilon) to prevent division by zero when taking the log.
        # The likelihood can be extremely small in certain cases, so without the epsilon, we may end up taking log(0)
        epsilon = 1e-9

        # Return the negative of the sum of the log-likelihoods
        return -np.sum(np.log(np.sum(likelihood_terms, axis=1) + epsilon))


    def fit(self, path: np.array, initial_params: List[float] = [0.1, 0.2, 1, 0.1, 0.2], max_iter: int = 1000, dt = 1/252,optimize_method: str = 'Nelder-Mead'):
        """
        Let's fit our MJD parameters to these returns by minimizing total negative log likelihood. Scipy.minimize has a number of algorithms, but we set Nelder-Mead as the default 
        given its robustness and its success in Furuit Tang's 2018 paper, 'Merton Jump Diffusion of Stock Price Data'

        Parameters:
        path: np.array representing a single path.
        initial_params: Initial guess for parameters. This is another magic number, which probably needs a better solution. It might be good to explore the convexity of this likelihood space analytically or through 
        analysis of some monte carlo of the parameters 

        Raises:
        ValueError: If optimization fails.
        """

        # Process the input path
        if self.optimize_method is None:
            self.optimize_method = optimize_method
        self.path = path
        self.S_last = self.path[-1]
        self.dt = dt  # Save dt for use in the callback

        # Optimize the parameters
        result = minimize(self.negative_log_likelihood, initial_params, method=self.optimize_method, options={'maxiter': max_iter}, callback=self._callback)


        # Handle unsuccessful optimization
        if result.success:
            self.mu, self.sigma, self.lambda_, self.mu_j, self.sigma_j = result.x
        else:
            raise ValueError("Optimization failed: " + result.message)


    def simulate(self, T: float, dt: float, num_sims: int, S_base: Optional[float] = None) -> np.array:
        """
        Simulates a Merton Jump Diffusion (MJD) process using provided parameters.

        Parameters:
        T: float, the total length of time for the simulation in years
        dt: float, the time-step size.
        num_sims: int, the number of simulations.
        S_base: float, optional, the last known stock price. If not provided, the last known stock price (self.S_last)
                of the instantiated object will be used.

        Returns:
        np.array, a 2D array of simulations. The first axis represents different simulations,
                  and the second axis represents time steps.
        """
        if S_base is None:
          S_base = self.S_last

        if T % dt != 0:
          warnings.warn(f"T = {T} is not a multiple of dt = {dt}. Simulating up to T = {T_new} instead.")

        t = np.arange(0, T, dt)
        dN = np.random.poisson(self.lambda_ * dt, (num_sims, len(t)))
        Y = self.mu_j * dN + self.sigma_j * np.sqrt(dN) * np.random.randn(num_sims, len(t))
        dW = np.sqrt(dt) * np.random.randn(num_sims, len(t))
        sim_steps = (self.mu - self.sigma**2 / 2) * dt + self.sigma * dW + Y
        S = np.empty((num_sims, len(t)))
        S[:, 0] = S_base * np.exp(sim_steps[:, 0])  # Generate the first price based on S_base
        S[:, 1:] = S_base * np.cumprod(np.exp(sim_steps[:, 1:]), axis=1)
        return S


    def predict(self, T: float, dt: float, num_sims: int, S_base: Optional[float] = None) -> Dict[str, Union[np.array, float]]:
        """
        Uses the simulate method to simulate paths of a Merton Jump Diffusion (MJD) process using the provided parameters. 
        Calculates and returns the mean, standard deviation of the final simulated stock prices, and the mean log return.

        Parameters:
        T: float, the total length of time for the simulation in years.
        dt: float, the time-step size.
        num_sims: int, the number of simulations.
        S_base: float, optional, the last known stock price. If not provided, the last known stock price (self.S_last)
                of the instantiated object will be used.

        Returns:
        dict: A dictionary containing the mean ('mean_price'), standard deviation ('std_dev_price') of the final stock prices across simulations,
              and the mean log return ('mean_log_return') across simulations.
        """

        if S_base is None:
            S_base = self.S_last

        if T % dt != 0:
            warnings.warn(f"T = {T} is not a multiple of dt = {dt}. Predicting up to T = {T_new} instead.")

        S = self.simulate(T, dt, num_sims, S_base)
        mean_log_return = np.mean(np.log(S[-1, :] / S[0, :]) / T)
        return {
            'mean_final_price': np.mean(S[:, -1]),
            'std_dev_final_price': np.std(S[:, -1]),
            'mean_log_return': mean_log_return,
        }


    def get_params(self):
        """
        Return a dictionary of the MJD model's parameters, including its optimization method, similar to sklearn's get_params or Tensorflow/Keras' get_config()

        Returns:
        A dictionary of the model's parameters
        """
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'lambda': self.lambda_,
            'mu_j': self.mu_j,
            'sigma_j': self.sigma_j,
            'optimize_method': self.optimize_method
        }
    def set_params(self, **kwargs):
        """
        Manually set some or all of the model's parameters

        Parameters:
        kwargs: Dictionary of parameter keys and their intended, new values
        """
        if 'mu' in kwargs:
            self.mu = kwargs['mu']
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        if 'lambda_' in kwargs:
            self.lambda_ = kwargs['lambda_']
        if 'mu_j' in kwargs:
            self.mu_j = kwargs['mu_j']
        if 'sigma_j' in kwargs:
            self.sigma_j = kwargs['sigma_j']
        if 'optimize_method' in kwargs:
            self.optimize_method = kwargs['optimize_method']
    def get_likelihood_history(self) -> List[Dict[str, Union[int, float]]]:
        """
        Return the list of dictionaries that represent the history of likelihood values
        and parameters during the optimization.

        Returns:
        A list of dictionaries.
        """
        return self.likelihood_values.copy()
