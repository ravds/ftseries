class GeometricBrownianMotion:
    """
    Create a geometric Brownian motion (GBM) process, a stochastic process popularized by Black and Scholes
    in their 1973 paper, "The Pricing of Options and Corporate Liabilities" based on the idea of random price vibrations 
    that remain after relevant information's immediate incorporation according to the strong efficient markets hypothesis.
    Both geometric and arithmetic Brownian motion are Markov processes, which mean that their future states depend only on their current states, and
    Levy proceses, which mean that increments over equal time intervals are independent and identically distributed. 


    Attributes:
    - path (np.array): A historical path 
    - mu (float): Expected return.
    - sigma (float): Volatility.
    - dt (float): Length of each time step.
    - bessel (bool): If True, apply Bessel's correction for sample variance calculation.
    - optimize_method (str): Method for optimization if numerical optimization is used.
    - likelihood_values (List[Dict[str, Union[int, float]]]): List of likelihood and parameters during optimization.

    Methods:
    - calculate_log_returns: Compute log returns from a given path.
    - _callback: A callback function to monitor optimization progress.
    - negative_log_likelihood: Computes the negative log-likelihood for given parameters.
    - fit: Fit the GBM parameters to a given path.
    - simulate: Simulate future stock price paths.
    - predict: Predict mean, standard deviation of final stock prices, and mean log return.
    - get_params: Return a dictionary of the GBM model's parameters.
    - set_params: Set the GBM model's parameters.
    - get_likelihood_history: Retrieve the likelihood and parameters history during optimization.
    """
    def __init__(self, path=None, mu=0, sigma=1, dt=1/252, bessel=True, optimize_method: str = None):
        self.path = path
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.bessel = bessel
        self.optimize_method = optimize_method
        self.likelihood_values = []

    def calculate_log_returns(self, path):
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
        })


    def negative_log_likelihood(self, params):
        """
        Function to calculate the total negative log-likelihood of a sequence of a log quotients
        Note: The log funciton is monotonic over its support, (0, inf), with the advantage that a log of a product is a sum of the logarithms of the product terms
        We will add a small epsilon to prevent passing any 0's to the logarithm, which is a fairly commmon appraoch

        Parameters:
        params: List of parameters [mu, sigma].

        Returns:
        Total negative log-likelihood.
        """
        # Unpack parameters
        mu, sigma = params

        dt = self.dt  # use saved dt
        steps = self.calculate_log_returns(self.path)  # use saved log returns

        likelihood_terms = norm.pdf(steps, loc=(mu - sigma**2/2)*dt, scale = np.sqrt(sigma**2*dt))

        # Add a small constant (epsilon) to prevent division by zero when taking the log.
        # The likelihood can be extremely small in certain cases, so without the epsilon, we may end up taking log(0)
        epsilon = 1e-9

        # Return the negative of the sum of the log-likelihoods
        return -np.sum(np.log(likelihood_terms, axis=1) + epsilon)

    def fit(self, path: np.array, initial_params: List[float] = [0, 1], max_iter: int = 1000, dt = 1/252, direct: bool = False, optimize_method: str = 'Nelder-Mead'):
        """
        Function to fit class attributes in-place/ to fit geometric brownian motion parameters to some history. There is a closed-form MLE, but I've added a numerical optimization option for exploratory purposes.

        Parameters:
        path: history (of prices usually)
        initial_params: guess/initialization for GBM params
        max_iter: maximum number of iterations that the numerical optimization algorithm will attempt before raising a valueerror
        dt: length of each time step
        direct: a boolean that dictates whether we use the closed-form or a numerical technique
        optimize_method: optimization method that we will pass to scipy's minimize function



        Returns:
        Nothing
        """

        if self.path is None:
            self.path = path
        log_returns = calculate_log_returns(path)
        ddof = 1 if self.bessel else 0

        if direct:
            self.mu = (np.mean(log_returns) + np.var(log_returns, ddof=ddof)/2) / dt
            self.sigma = np.sqrt(np.var(log_returns, ddof=ddof) / dt)
        else:
            result = minimize(self.negative_log_likelihood, initial_params, method=optimize_method, options={'maxiter': max_iter})

            if result.success:
                self.mu, self.sigma = result.x
            else:
                raise ValueError("Optimization failed: " + result.message)

    def simulate(self, T: float = 1, dt_sim: float = 1/252, num_sims: int = 100, S_base: Optional[float] = None) -> np.array:
        """
        Simulates a Geometric Brownian Motion (GBM) process using provided parameters.

        Parameters:
        T: float, the total length of time for the simulation in years
        dt_sim: float, the time-step size.
        num_sims: int, the number of simulations.
        S_base: float, optional, the last known stock price. If not provided, the last known stock price
                of the instantiated object will be used.

        Returns:
        np.array, a 2D array of simulations. The first axis represents different simulations,
                  and the second axis represents time steps
        """
        if S_base is None:
            S_base = self.path[-1] if self.path is not None else 100
        t = np.arange(0, T, dt_sim)
        sim_steps = (self.mu - 0.5 * self.sigma**2) * dt_sim + self.sigma * np.sqrt(dt_sim) * np.random.normal(size=(num_sims, len(t)))
        S = S_base * np.cumprod(np.exp(sim_steps), axis=1)
        return S

    def predict(self, dt_sim=1/252, T=1, num_sim=100, S0=None) -> Dict[str, Union[np.array, float]]:
        """
        Uses the simulate method to simulate paths of a Geometric Brownian Motion (GBM) process using the provided parameters.
        Calculates and returns the mean, standard deviation of the final simulated stock prices, and the mean log return.

        Parameters:
        dt_sim: float, the time-step size. Default is 1/252, which corresponds to daily steps.
        T: float, the total length of time for the simulation in years. Default is 1.
        num_sim: int, the number of simulations. Default is 100.
        S0: float, optional, the initial stock price. If not provided, the last known stock price (self.path[-1])
            of the instantiated object will be used.

        Returns:
        dict: A dictionary containing the mean ('mean_final_price'), standard deviation ('std_dev_final_price') of the final stock prices across simulations,
              and the mean log return ('mean_log_return') across simulations.
        """

        if S0 is None:
            S0 = self.S_last

        if T % dt_sim != 0:
            warnings.warn(f"T = {T} is not a multiple of dt_sim = {dt_sim}. Predicting up to T = {T_new} instead.")

        S = self.simulate(dt_sim, T, num_sim, S0)
        return {
            'mean_final_price': np.mean(S[:, -1]),
            'std_dev_final_price': np.std(S[:, -1]),
            'mean_log_return': np.mean(np.log(S[:, -1] / S[:, 0]) / T),
        }


    def get_params(self):
        """
        Return a dictionary of the GBM model's parameters, including its optimization method, similar to sklearn's get_params or Tensorflow/Keras' get_config()

        Returns:
        A dictionary of the model's parameters
        """
        return {
            'mu': self.mu,
            'sigma': self.sigma,
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