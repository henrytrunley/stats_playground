import argparse
import matplotlib.pyplot as plt
import numpy as np

from distributions import Gaussian, Pareto


if __name__=='__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-n', '--sample_size', type=int, required=True,
                                 help='sample size to infer mean from')
    argument_parser.add_argument('-k', '--experiments', type=int, required=True,
                                 help='num experiments to infer mean for')
    argument_parser.add_argument('-a', '--alpha', type=float, required=True,
                                 help='alpha (shape) parameter of pareto distribution')
    argument_parser.add_argument('-m', '--mean', type=float, required=True,
                                 help='mean of distribution')
    args = argument_parser.parse_args()

    sample_size = args.sample_size
    experiments = args.experiments
    mean = args.mean

    # pareto params
    alpha = args.alpha  # corresponds to 80/20 principle
    x_m = mean * (alpha - 1) / alpha

    pareto = Pareto(x_m, alpha, min_prob = 1/(sample_size*100))

    plt.figure(figsize=(10, 10))
    all_data = pareto.sample(sample_size * experiments)
    data = np.reshape(all_data, (experiments, sample_size))
    experiment_means = data.mean(axis=1)
    plt.hist(experiment_means, bins=experiments//10)
    plt.vlines(x=mean, ymin=0, ymax=50, color='red')
    plt.show()