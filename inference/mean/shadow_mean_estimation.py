import argparse
import matplotlib.pyplot as plt
import numpy as np

from distributions import Pareto


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
    alpha = args.alpha  # 1.16 corresponds to 80/20 principle
    x_m = mean * (alpha - 1) / alpha

    pareto = Pareto(x_m, alpha, min_prob = 1/(sample_size*100))

    all_data = pareto.sample(sample_size * experiments)
    data = np.reshape(all_data, (experiments, sample_size))
    experiment_means = data.mean(axis=1)
    max_likelihood_x_m = data.min(axis=1)
    max_likelihood_alpha = sample_size / np.log(data / np.array([max_likelihood_x_m]).T).sum(axis=1)
    max_likelihood_mean = max_likelihood_alpha * max_likelihood_x_m / (max_likelihood_alpha - 1)

    min_mean = min(experiment_means.min(), max_likelihood_mean.min())
    max_mean = min(experiment_means.max(), max_likelihood_mean.max())
    bins = np.linspace(min_mean, max_mean, int(np.sqrt(experiments)))

    plt.figure(figsize=(10, 10))
    plt.hist(experiment_means, bins=bins, label='actual mean', alpha=.5)
    plt.hist(max_likelihood_mean, bins=bins, label='ML mean', alpha=.5)
    plt.vlines(x=mean, ymin=0, ymax=50, color='red')
    plt.legend()
    plt.show()