import matplotlib.pyplot as plt

from distributions import *

if __name__=="__main__":
    stable = Stable(2, 0, 0, 1/np.sqrt(2))
    normal = Gaussian(0, 1)

    # plt.figure(figsize=(10, 10))
    # plt.hist(stable.sample(1000), bins=100)
    # plt.show()

    x = np.linspace(-10, 10, 10000)
    # cdf = stable.cdf(x)
    #
    # plt.figure(figsize=(10, 10))
    # plt.plot(x, cdf, label='stable CDF')
    #
    # plt.figure(figsize=(10, 10))
    # x = np.linspace(.01, .99, 1000)
    # ppf = stable.ppf(x)
    # plt.plot(x, ppf, label='stable PPF')

    plt.figure(figsize=(10,10))
    pdf = stable.pdf(x)
    plt.plot(x, pdf, label='stable')

    pdf = normal.pdf(x)
    plt.plot(x, pdf, label='gaussian')

    plt.show()
    # plt.xlim(-10, 10)
    # plt.legend()
    # plt.show()

    # df = f[1] - f[0]
    # x = np.fft.fftfreq(len(f), d=df/(2*np.pi))
    # y = np.fft.ifft(stable.characteristic_function(np.fft.fftshift(f)))
    #
    # plt.figure(figsize=(10, 10))
    # plt.plot(x, y)
    # plt.title('stable dist')
    #
    # plt.show()

