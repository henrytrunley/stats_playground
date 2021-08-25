import numpy as np

class Distribution:
    """
    Base class for all probability distributions to come.

    Records a numerical distribution after first use for linear interpolation to avoid time consuming calculations.

    Subclasses of Distribution should define a self.construct_discrete_pdf method to be usable.

    self.construct_discrete_pdf requirements:
        Positional arguments:
            x -- a 1d numpy array of coordinates to evaluate the PDF at
        Outputs:
            a numpy array the same shape as x containing the corresponding PDF values
    """

    sample_datapoints = 10**7

    def __init__(self, min_x: float, max_x: float):
        self.min_x = min_x
        self.max_x = max_x

    def sample(self, n: int):
        return self.ppf(np.random.uniform(size=n, low=0., high=1.))

    def cdf(self, x: np.array):
        self._ensure_discrete_cdf()
        return self._interpolate_discrete_cdf(x)

    def pdf(self, x: np.array):
        self._ensure_discrete_pdf()
        return self._interpolate_discrete_pdf(x)

    def ppf(self, x: np.array):
        self._ensure_discrete_cdf()
        return self._interpolate_discrete_ppf(x)

    def construct_discrete_pdf(self, x: np.array):
        raise AttributeError('No user defined method "construct_discrete_pdf". See docstring for usage.')

    def _ensure_discrete_pdf(self):
        if not hasattr(self, '_discrete_pdf'):
            self._discrete_x_coords = np.linspace(self.min_x, self.max_x, self.sample_datapoints)
            self._dx = self._discrete_x_coords[1] - self._discrete_x_coords[0]
            self._discrete_pdf = self.construct_discrete_pdf(self._discrete_x_coords)

    def _ensure_discrete_cdf(self):
        self._ensure_discrete_pdf()
        if not hasattr(self, '_discrete_cdf'):
            self._discrete_cdf = np.cumsum(self._discrete_pdf * self._dx)
            self._discrete_cdf /= np.max(self._discrete_cdf)

    def _interpolate_discrete_cdf(self, x: np.array):
        return np.interp(
            x=x,
            xp=self._discrete_x_coords,
            fp=np.abs(self._discrete_cdf),
            left=0.,
            right=1.,
        )

    def _interpolate_discrete_pdf(self, x: np.array):
        return np.interp(
            x=x,
            xp=self._discrete_x_coords,
            fp=np.abs(self._discrete_pdf),
            left=0.,
            right=0.,
        )

    def _interpolate_discrete_ppf(self, x: np.array):
        return np.interp(
            x=x,
            xp=np.abs(self._discrete_cdf),
            fp=self._discrete_x_coords,
            left=self._discrete_x_coords[0],
            right=self._discrete_x_coords[-1],
        )

class CharacteristicFunctionBased(Distribution):
    """
    Intermediate class based off distribution that calculated the pdf from a user defined characteristic function.
    """

    def construct_discrete_pdf(self, x: np.array):
        n = len(x)
        dx = x[1] - x[0]
        f = np.fft.fftfreq(n, dx) * 2 * np.pi
        cf = self.characteristic_function(f)
        # return np.abs(np.fft.fftshift(np.fft.ifft(cf)) / self._dx)
        return np.fft.fftshift(np.fft.ifft(cf)) / dx

    def characteristic_function(self, f: np.array):
        raise AttributeError('No user defined method "characteristic_function". See docstring for usage.')


class Stable(CharacteristicFunctionBased):
    def __init__(self, alpha: float, beta: float, loc:float, scale:float):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.loc = loc

        x_width = 50 * scale

        super().__init__(min_x=loc - x_width, max_x=loc + x_width)

    def characteristic_function(self, f: np.array):
        return np.exp(
            1j * f * self.loc - np.abs(self.scale * f) ** self.alpha * (1 - 1j * self.beta * np.sign(f) * self.phi(f))
        )

    def phi(self, f):
        if self.alpha == 1:
            return - np.log(np.abs(self.scale * f)) * 2 / np.pi
        else:
            return np.tan(np.pi * self.alpha / 2)


class Cauchy(Stable):
    def __init__(self, x0: float, gamma: float):
        self.x0 = x0
        self.gamma = gamma
        super().__init__(1., 0., x0, gamma)

    def construct_discrete_pdf(self, x: np.array):
        return 1 / (np.pi * self.gamma * (1 + ((x - self.x0) / self.gamma)**2))


class Gaussian(Stable):
    def __init__(self, mu: float, sigma: float):
        super().__init__(2., 0., mu, sigma/np.sqrt(2))
        self.sigma = sigma

    def construct_discrete_pdf(self, x: np.array):
        return np.exp(- ((x - self.loc) / self.sigma)**2 / 2) / (np.sqrt(2*np.pi) * self.sigma)


class Levy(Stable):
    def __init__(self, mu: float, c: float):
        self.mu = mu
        self.c = c
        super().__init__(.5, 1., mu, c)

    def construct_discrete_pdf(self, x: np.array):
        return np.sqrt(self.c / (2 * np.pi)) * np.exp(-self.c / (2 * x - self.mu)) / (x - self.mu)**(3/2)

class Empirical(CharacteristicFunctionBased):
    def __init__(self, data: np.array):
        self.data = data
        self.loc = np.median(self.data)
        self.scale = np.quantile(data, .75) - np.quantile(data, .25)

        x_width = 50 * self.scale

        super().__init__(min_x=self.loc - x_width, max_x=self.loc + x_width)

    def characteristic_function(self, f: np.array):
        return np.mean(np.exp(1j * np.array([self.data]).T * np.array([f])), axis=0)

class Pareto(Distribution):
    def __init__(self, x_m: float, alpha: float, min_prob: float = 10**-6):
        self.x_m = x_m
        self.alpha = alpha
        max_x = x_m / min_prob**(1/alpha)
        super().__init__(min_x=x_m, max_x=max_x)

    def construct_discrete_pdf(self, x: np.array):
        return (x >= self.x_m) * self.alpha * self.x_m**self.alpha / x**(self.alpha + 1)