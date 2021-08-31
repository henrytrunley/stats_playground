import numpy as np
import matplotlib.pyplot as plt
import argparse

from distributions import Pareto

if __name__=="__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-n', '--sample_size', type=int, required=True,
                                 help='sample size to infer mean from')
    argument_parser.add_argument('-a', '--alpha', type=float, required=True,
                                 help='alpha (shape) parameter of pareto distribution')
    argument_parser.add_argument('-m', '--mean', type=float, required=True,
                                 help='mean of distribution')
    args = argument_parser.parse_args()

    sample_size = args.sample_size
    mean = args.mean

    # pareto params
    alpha = args.alpha  # corresponds to 80/20 principle
    x_m = mean * (alpha - 1) / alpha

    pareto = Pareto(x_m, alpha, min_prob= 1 / (sample_size * 100))

    data = pareto.sample(sample_size)

    log_data = np.log(np.abs(np.sort(data)))
    log_percentile = np.log(np.linspace(len(data), 1, len(data)) / len(data))

    x = np.linspace(x_m, data.max(), 10000)
    y = pareto.sf(x)
    log_x = np.log(x)
    log_y = np.log(y)

    plt.figure(figsize=(10, 10))
    plt.scatter(log_data, log_percentile, color='blue')
    plt.plot(log_x, log_y, color='red')
    plt.show()

    # log_x = np.log(np.abs(x[mask]))
    # log_y = np.log(np.abs(y[mask]))
    #
    # X = np.array([log_x])
    # X = np.vstack([np.ones((1, X.shape[1])), X]).T
    # Y = np.array([log_y]).T
    # beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    # m = beta[1][0]
    # print(-m)
    #
    # plt.figure(figsize=(10,10))
    # # plt.plot(x, empirical.cdf(x), label='sample')
    # plt.plot(log_x, log_y, label='population')
    # plt.title('CDF')
    # plt.legend()
    # plt.show()




