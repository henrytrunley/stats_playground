import numpy as np

class Distribution:
    sample_datapoints = 10 ** 6

    def __init__(self):
        pass

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

    def _ensure_discrete_pdf(self):
        if not hasattr(self, '_discrete_pdf'):
            self._discrete_x_coords = np.linspace(self.loc - self.x_width, self.loc + self.x_width, self.sample_datapoints)
            self._dx = self._discrete_x_coords[1] - self._discrete_x_coords[0]
            n = len(self._discrete_x_coords)
            f = np.fft.fftfreq(n, self._dx) * 2 * np.pi
            cf = self.characteristic_function(f)
            # self._discrete_pdf = np.abs(np.fft.fftshift(np.fft.ifft(cf)) / self._dx)
            self._discrete_pdf = (np.fft.fftshift(np.fft.ifft(cf)) / self._dx)

    def _ensure_discrete_cdf(self):
        self._ensure_discrete_pdf()
        if not hasattr(self, '_discrete_cdf'):
            self._discrete_cdf = np.cumsum(self._discrete_pdf * self._dx)

    def _interpolate_discrete_cdf(self, x: np.array):
        return np.interp(x, self._discrete_x_coords, np.abs(self._discrete_cdf))

    def _interpolate_discrete_pdf(self, x: np.array):
        return np.interp(x, self._discrete_x_coords, np.abs(self._discrete_pdf))

    def _interpolate_discrete_ppf(self, x: np.array):
        return np.interp(x, np.abs(self._discrete_cdf), self._discrete_x_coords)

class Stable(Distribution):
    def __init__(self, alpha: float, beta: float, loc:float, scale:float):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.loc = loc

        self.x_width = 50 * scale

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
    def __init__(self, x0: float, scale: float):
        super().__init__(1., 0., x0, scale)

    def pdf(self, x: np.array):
        pass


class Gaussian(Stable):
    def __init__(self, mu: float, sigma: float):
        super().__init__(2., 0., mu, sigma/np.sqrt(2))
        self.sigma = sigma

    def pdf(self, x: np.array):
        return np.exp(- ((x - self.loc) / (self.sigma))**2 / 2) / (np.sqrt(2*np.pi) * self.sigma)


class Levy(Stable):
    def __init__(self, mu: float, c: float):
        super().__init__(.5, 1., mu, c)

    def pdf(self, x: np.array):
        pass

class Empirical(Distribution):
    def __init__(self, data):
        self.data = data
        self.loc = np.median(self.data)
        self.scale = np.quantile(data, .75) - np.quantile(data, .25)
        self.x_width = 20 * self.scale

    def characteristic_function(self, f: np.array):
        return np.mean(np.exp(1j * np.array([self.data]).T * np.array([f])), axis=0)
