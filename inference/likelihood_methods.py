import numpy as np
from scipy.optimize import minimize

class LikelihoodMethods:
    def __init__(self, data: np.array, pdf: callable):
        self.data = data
        self.pdf = pdf

    def log_likelihood(self, params):
        return np.log(pdf(x, *params)).sum()

    def cost_function(self, params):
        return -self.log_likelihood(params)

    def maximise_likelihood(self, initial_params):
        return minimize(self.cost_function, initial_params)