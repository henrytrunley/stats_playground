import argparse
import matplotlib.pyplot as plt
import numpy as np

from distributions import Gaussian, Pareto


if __name__=='__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-n', '--sample_size', type=int, required=True,
                                 help='max sample size to infer mean from')
    argument_parser.add_argument('-a', '--alpha', type=float, required=True,
                                 help='alpha (shape) parameter of pareto distribution')
    argument_parser.add_argument('-m', '--mean', type=float, required=True,
                                 help='mean of distributions')
    args = argument_parser.parse_args()

    sample_size = args.sample_size
    mean = args.mean

    # pareto params
    alpha = 1.16  # corresponds to 80/20 principle
    x_m = mean * (alpha - 1) / alpha

    # gaussian params
    mu = mean
    sigma = 1

    distributions = {}
    distributions['normal'] = Gaussian(mu,sigma)
    distributions['pareto'] = Pareto(x_m, alpha)

    plt.figure(figsize=(10, 10))
    for name, dist in distributions.items():
        data = dist.sample(sample_size)
        n = np.arange(1, len(data)+1, 1)
        cumulative_mean = np.cumsum(data) / n
        plt.plot(n, cumulative_mean, label=name)
    plt.legend()
    plt.show()