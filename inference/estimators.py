import numpy as np
from scipy.optimize import minimize
import typing


class Likelihood:
    def __init__(self, data: np.array, pdf: callable):
        self.data = data
        self.pdf = pdf

    def log_likelihood(self, params):
        return np.log(pdf(x, *params)).sum()

    def cost_function(self, params):
        return -self.log_likelihood(params)

    def maximise_likelihood(self, initial_params):
        return minimize(self.cost_function, initial_params)


def pareto_max_likelihood_params(data):
    if data.ndim == 1:
        data = np.array([data])
    sample_size = data.shape[1]
    max_likelihood_x_m = data.min(axis=1)
    max_likelihood_alpha = sample_size / np.log(data / np.array([max_likelihood_x_m]).T).sum(axis=1)
    return max_likelihood_x_m, max_likelihood_alpha

def log_survival_function(data: np.array) -> typing.Tuple[np.array, np.array]:
    log_data = np.log(np.abs(np.sort(data)))
    log_percentile = np.log(np.linspace(len(data), 1, len(data)) / len(data))
    return log_data, log_percentile

def alpha_from_log_sf(log_data: np.array, log_sf: np.array) -> float:
    beta = OLS_1D(log_data, log_sf)
    m = beta[1][0]
    alpha = -m
    return alpha

def OLS_1D(x: np.array, y: np.array):
    X = np.array([x])
    X = np.vstack([np.ones((1, X.shape[1])), X]).T
    Y = np.array([y]).T
    return OLS(X, Y)

def OLS(X: np.array, Y: np.array):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return beta