from astropy.convolution import Box1DKernel
import emcee
import numpy as np
from scipy import integrate, optimize, signal

from .utils import acf, get_noise, smooth


def Vrng_Basri2011(y):
    """Basri et al. 2011, AJ, 141, 20"""
    vrng = np.percentile(y, 95) - np.percentile(y, 5)
    return vrng


def HFrms_Basri2011(y):
    """Basri et al. 2011, AJ, 141, 20"""
    yf = smooth(y, Box1DKernel(width=4))
    resid = y - yf
    hfrms = np.std(resid)
    return hfrms


def Sph_Mathur2014(y, dy=None):
    """Mathur et al. 2014, A&A, 562, 124"""
    if dy is None:
        noise = get_noise(y)
    else:
        noise = np.min(dy)
    sph = np.sqrt(np.std(y) ** 2 - noise ** 2)
    return sph


def Sph_timeseries_Mathur2014(y, t, prot, dy=None, k=5):
    """Mathur et al. 2014, A&A, 562, 124"""
    window = k * prot
    mint = np.min(t)
    maxt = np.max(t)
    t_sph, y_sph = [], []
    N = int(max(1, (maxt - mint) // window))
    for i in range(N):
        indices = (t >= mint + i * window) & (t < mint + (i + 1) * window)
        yi = y[indices]
        if dy is not None:
            dyi = dy[indices]
        else:
            dyi = None
        try:
            y_sph = np.append(y_sph, Sph_Mathur2014(yi, dyi))
            t_sph = np.append(t_sph, mint + (i + 0.5) * window)
        except ValueError:
            continue
    t_sph = t_sph[np.isfinite(y_sph)]
    y_sph = y_sph[np.isfinite(y_sph)]
    return t_sph, y_sph


def Contrast_Mathur2014(y, t, prot, dy=None, k=5):
    """Mathur et al. 2014, A&A, 562, 124"""
    to, yo = Sph_timeseries_Mathur2014(y, t, prot, dy, k)
    sph = Sph_Mathur2014(y, dy)
    nlo = yo < sph
    nhi = ~nlo
    C = yo[nhi].mean() / yo[nlo].mean()
    return C


def m_k(y, t, prot, dy=None, k=5):
    to, yo = Sph_timeseries_Mathur2014(y, t, prot, dy, k)
    return yo.std() / yo.mean()


def Reff_He2015(y):
    """He et al. 2015, ApJS, 221, 18"""
    reff = np.sqrt(8) * np.std(y)
    return reff


def iAC_He2015(y):
    """He et al. 2015, ApJS, 221, 18"""
    N = y.size // 2
    lags, ryy = acf(y, maxlag=N)
    iac = integrate.simps(np.abs(ryy), lags) / N
    return iac


def SDR_Basri2018(t, y, prot):
    """Basri & Nguyen 2018, ApJ, 863, 190"""
    if prot > 8:
        yf = smooth(y, Box1DKernel(width=prot / 8))
    else:
        yf = y.copy()
    dips, _ = signal.find_peaks(-yf, width=4)
    seps = np.diff(t[dips])
    singledips = seps > 0.75 * prot
    doubledips = ~singledips
    singlemode = seps[singledips].sum()
    doublemode = seps[doubledips].sum()
    sdr = np.log10(singlemode / doublemode)
    return sdr


def tau_ACF_Santos2021(lags, ryy, prot, random_seed=None, use_prior=False):
    """Santos et al. 2021"""
    rng = np.random.default_rng(random_seed)
    maxima, _ = signal.find_peaks(ryy)
    idx = np.where(np.diff(np.diff(np.sign(np.diff(ryy[maxima])))) == -2)[0][0] + 2
    max_lag = maxima[idx]
    nflux_j = lags.size - np.arange(max_lag)
    ryy = ryy[:max_lag]
    lags = lags[:max_lag]
    sigma_p = 0.2 * prot
    sigma_j = 1 / np.sqrt(nflux_j)

    def predict(theta):
        tau, a, b, y0, P = theta
        phi = 2 * np.pi * lags / P
        return (1 - lags / tau) * (a * np.cos(phi) + b * np.cos(2 * phi) + y0)

    def log_prior(theta):
        tau, a, b, y0, P = theta
        log_tau = np.log10(tau)
        if not (
            (-1 < log_tau < 3)
            and (0.01 < a < 1.5)
            and (0.001 < b < 1)
            and (-1 < y0 < 1)
        ):
            return -np.inf
        chi = ((P - prot) / sigma_p) ** 2
        norm_const = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma_p)
        lp = norm_const - 0.5 * chi
        return lp

    def nll(theta):
        y = predict(theta)
        chi = np.sum(np.square((ryy - y) / sigma_j))
        norm_const = -0.5 * max_lag * np.log(2 * np.pi) - 0.5 * np.log(sigma_j).sum()
        return 0.5 * chi - norm_const

    def logprob(theta):
        return log_prior(theta) - nll(theta)

    n_walkers = 25
    n_dim = 5
    n_steps = 10_000
    burn = 8_000
    if use_prior:
        p0 = np.vstack(
            [
                10 ** rng.uniform(-1, 3, n_walkers),
                rng.uniform(0.01, 1.5, n_walkers),
                rng.uniform(0.001, 1, n_walkers),
                rng.uniform(-1, 1, n_walkers),
                rng.normal(prot, sigma_p, n_walkers),
            ]
        ).T
    else:
        x0 = [5 * prot, 0.75, 0.5, 0.0, prot]
        bounds = [
            (0.1, 1000),
            (0.01, 1.5),
            (0.001, 1),
            (-1, 1),
            (0.2 * prot, 1.8 * prot),
        ]
        res = optimize.minimize(fun=nll, x0=x0, method="L-BFGS-B", bounds=bounds)
        p0 = res.x + 1e-2 * rng.standard_normal((n_walkers, n_dim))
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, logprob)
    sampler.run_mcmc(p0, n_steps, progress=True)
    samples = sampler.get_chain(discard=burn, flat=True)
    tau = np.percentile(samples[:, 0], [16, 50, 84], axis=0)
    theta_MAP = np.median(samples, axis=0)
    reduced_chi = np.sum(np.square((ryy - predict(theta_MAP)) / sigma_j)) / (
        max_lag - n_dim
    )
    return tau, theta_MAP, reduced_chi
