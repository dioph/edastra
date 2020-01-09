import numpy as np
from astropy.convolution import Box1DKernel
from scipy import integrate, signal

from .utils import smooth, get_noise, acf


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
        yf = smooth(y, Box1DKernel(width=prot/8))
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
