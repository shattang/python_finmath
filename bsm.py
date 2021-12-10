import scipy.stats as spst
import numpy as np


def d1(S, K, r, sigma, T):
    return (np.log(S/K) + (r+sigma*sigma/2)*T)/(sigma*np.sqrt(T))


def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma*np.sqrt(T)


def norm_cdf(x):
    return spst.norm.cdf(x)


def norm_pdf(x):
    return spst.norm.pdf(x)


def call_price(S, K, r, sigma, T):
    return np.maximum(S - K, 0) if T == 0 else\
        S*norm_cdf(d1(S, K, r, sigma, T)) - K*np.exp(-r*T) * \
        norm_cdf(d2(S, K, r, sigma, T))


def call_delta(S, K, r, sigma, T):
    return norm_cdf(d1(S, K, r, sigma, T))


def gamma(S, K, r, sigma, T):
    return norm_pdf(d1(S, K, r, sigma, T))/(S*sigma*np.sqrt(T))


def vega(S, K, r, sigma, T):
    return S*norm_pdf(d1(S, K, r, sigma, T))*np.sqrt(T)


def call_theta(S, K, r, sigma, T):
    aux1 = -S*norm_pdf(d1(S, K, r, sigma, T))*sigma/(2*np.sqrt(T))
    aux2 = -r*K*np.exp(-r*T)*norm_cdf(d2(S, K, r, sigma, T))
    return aux1+aux2


def call_rho(S, K, r, sigma, T):
    return K*T*np.exp(-r*T)*norm_cdf(d2(S, K, r, sigma, T))


def put_price(S, K, r, sigma, T):
    return np.maximum(K-S, 0) if T == 0 else\
        K*np.exp(-r*T)*norm_cdf(-1*d2(S, K, r, sigma, T)) - \
        S*norm_cdf(-1*d1(S, K, r, sigma, T))


def put_delta(S, K, r, sigma, T):
    return norm_cdf(d1(S, K, r, sigma, T)) - 1


def put_theta(S, K, r, sigma, T):
    aux1 = -S*norm_pdf(d1(S, K, r, sigma, T))*sigma/(2*np.sqrt(T))
    aux2 = r*K*np.exp(-r*T)*norm_cdf(-1*d2(S, K, r, sigma, T))
    return aux1+aux2


def put_rho(S, K, r, sigma, T):
    return -K*T*np.exp(-r*T)*norm_cdf(-1*d2(S, K, r, sigma, T))
