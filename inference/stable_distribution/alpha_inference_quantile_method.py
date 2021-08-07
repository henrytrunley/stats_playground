import numpy as np
import matplotlib.pyplot as plt
import argparse

from distributions import Stable, Empirical


def alpha_from_quantiles(data):
    q5, q25, q50, q75, q95 = np.quantle(data, [.05, .25, .5, .75, .95])
    nu_alpha = (q95 - q5) / (q75 - q25)
    nu_beta = (q95 - 2*q50 + q5) / (q95 - q5)
    return nu_alpha, nu_beta

if __name__=="__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-n', '--sample_size', type=int, required=True, help='number of Monte-Carlo data points to infer parameters from')
    argument_parser.add_argument('-alpha', '--alpha', type=float, required=True, help='alpha (shape) parameter of stable distribution')
    argument_parser.add_argument('-beta', '--beta', type=float, default=0., help='beta (skew)  parameter of stable distribution')
    argument_parser.add_argument('-loc', '--loc', type=float, default=0., help='location parameter of stable distribution')
    argument_parser.add_argument('-scale', '--scale', type=float, default=1., help='scale parameter of stable distribution')
    args = argument_parser.parse_args()
    
    n = args.sample_size
    alpha = args.alpha
    beta = args.beta
    loc = args.loc
    scale = args.scale

    stable = Stable(alpha, beta, loc, scale)

    # measured_alphas = []
    #
    # for i in range(300):
    #     print(i)
    #     data = stable.sample(n)
    #     tail_n = len(data) // 10
    #     log_x = np.log(np.abs(np.sort(data)))[-tail_n:]
    #     log_rank = np.log(np.linspace(len(data), 1, len(data)))[-tail_n:]
    #
    #     X = np.array([log_x])
    #     X = np.vstack([np.ones((1, X.shape[1])), X]).T
    #     Y = np.array([log_rank]).T
    #     beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    #     m = beta[1][0]
    #     alpha_m = (m - 1) / 2
    #
    #     measured_alphas.append(alpha_m)
    # plt.figure(figsize=(10, 10))
    # plt.hist(np.array(measured_alphas), bins=len(measured_alphas) // 10)
    # plt.show()

    # empirical = Empirical(data)
    #
    x = np.linspace(-20,20,10000)
    y = 1 - stable.cdf(x)
    mask = (y < .1) & (y > .01)
    log_x = np.log(np.abs(x[mask]))
    log_y = np.log(np.abs(y[mask]))

    X = np.array([log_x])
    X = np.vstack([np.ones((1, X.shape[1])), X]).T
    Y = np.array([log_y]).T
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    m = beta[1][0]
    print(-m)

    plt.figure(figsize=(10,10))
    # plt.plot(x, empirical.cdf(x), label='sample')
    plt.plot(log_x, log_y, label='population')
    plt.title('CDF')
    plt.legend()
    plt.show()
    #
    # plt.figure(figsize=(10, 10))
    # plt.plot(np.log(np.abs(x)**np.sign(x)), np.log(stable.pdf(x)), label='sample')
    # plt.plot(x, stable.pdf(x), label='population')
    # plt.hist(data, bins=50, label='sample', density=True)
    # plt.title('PDF')
    # plt.legend()



    #
    # X = np.array([log_x])
    # X = np.vstack([np.ones((1,X.shape[1])), X]).T
    # Y = np.array([log_rank]).T
    # beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    # m = beta[1][0]
    # alpha_m = - m - 1
    # print(alpha, alpha_m)
    #
    # plt.plot(log_x, log_x * beta[1] + beta[0])
    # plt.show()




