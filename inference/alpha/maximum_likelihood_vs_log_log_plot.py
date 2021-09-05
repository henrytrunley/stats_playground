import argparse
import matplotlib.pyplot as plt
import numpy as np

from distributions import Pareto
from inference.estimators import log_survival_function, alpha_from_log_sf, pareto_max_likelihood_params

if __name__ == "__main__":
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

    pareto = Pareto(x_m, alpha, min_prob=1 / (sample_size * 100))

    all_data = pareto.sample(sample_size * experiments)
    data = np.reshape(all_data, (experiments, sample_size))

    _, ml_alphas = pareto_max_likelihood_params((data))

    log_sf_alphas = []
    for i in range(experiments):
        experiment_data = data[i,:]
        log_data, log_sf = log_survival_function(experiment_data)
        log_sf_alpha = alpha_from_log_sf(log_data, log_sf)
        log_sf_alphas.append(log_sf_alpha)
    log_sf_alphas = np.array(log_sf_alphas)

    plt.figure(figsize=(10,10))
    plt.scatter(ml_alphas, log_sf_alphas)
    plt.plot(ml_alphas, ml_alphas, color='red')
    plt.ylabel('log_survival_function')
    plt.xlabel('max_likelihood')

    min_alpha = min(ml_alphas.min(), log_sf_alphas.min())
    max_alpha = min(ml_alphas.max(), log_sf_alphas.max())
    bins = np.linspace(min_alpha, max_alpha, int(np.sqrt(experiments)))
    plt.figure(figsize=(10,10))
    plt.hist(ml_alphas, bins=bins, density=True, alpha=.5, label='max_likelihood')
    plt.hist(log_sf_alphas, bins=bins, density=True, alpha=.5, label='log_survival_function')
    plt.legend()

    plt.show()