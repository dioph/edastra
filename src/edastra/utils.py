import os
import subprocess

from astropy.convolution import Gaussian1DKernel
from astropy.io import fits
import numpy as np
from scipy import interpolate, signal


def prot_acf(t, y, max_per=None, s=0, fill=True, criterium="height", **kwargs):
    if max_per is not None:
        max_per = float(max_per)
    lags, ryy = acf(y, t, s=s, fill=fill, maxlag=max_per)
    peaks, properties = signal.find_peaks(ryy, **kwargs)
    if criterium == "height":
        metric = ryy[peaks]
    elif criterium == "prominence":
        metric = properties["prominences"]
    bp = lags[peaks][metric.argmax()]
    return bp


def acf(y, t=None, maxlag=None, s=0, fill=False):
    """Auto-Correlation Function implemented using IFFT of the power spectrum.

    Parameters
    ----------
    y: array-like
        discrete input signal
    t: array-like (optional)
        time array
    maxlag: int or float (optional)
        Maximum lag to compute ACF.
        If given as a float, will be assumed to be a measure of time and the
        ACF will be computed for lags lower than or equal to `maxlag`.
    s: int (optional)
        standard deviation of Gaussian filter used to smooth ACF, measured in samples
    fill: bool (optional default=False)
        whether to use linear interpolation to sample signal uniformly

    Returns
    -------
    lags: array-like
        array of lags
    ryy: array-like
        ACF of input signal
    """
    if t is None:
        t = np.arange(len(y))
    if fill:
        t, y = fill_gaps(t, y)
    n = len(y)
    if maxlag is None:
        maxlag = n
    if type(maxlag) is float:
        maxlag = np.where(t - np.min(t) <= maxlag)[0][-1] + 1
    f = np.fft.fft(y - y.mean(), n=2 * n)
    ryy = np.fft.ifft(f * np.conjugate(f))[:maxlag].real
    if s > 0:
        ryy = smooth(ryy, Gaussian1DKernel(stddev=s))
    ryy /= ryy[0]
    lags = t[:maxlag] - np.min(t[:maxlag])
    return lags, ryy


def butterfly(
    t_start=0,
    t_sim=1000,
    act=1.0,
    pcyc=1.0,
    pover=0.0,
    lmin=0.0,
    lmax=70.0,
    random_seed=42,
    randomize=False,
):
    """Starspot Emergence Model: Butterfly Diagram

    Parameters
    ----------
    act: scalar
        Stellar magnetic activity with respect to the Sun.
    pcyc: scalar
        Magnetic activity cycle period in years.
    pover: scalar
        Overlap between consecutive cycles, in years.
    lmin, lmax: scalar
        Minimum/maximum possible latitude of spot emergence.
    randomize: bool
        Whether the spots emerge in random latitudes or follow a
        butterfly-like distribution.

    Returns
    -------
    tmax: ndarray
        Times of peak spot areas
    lats: ndarray
        Spot latitudes (radians)
    lons: ndarray
        Spot longitudes (radians)
    bmax: ndarray
        Peak spot field intensities (gauss)

    References
    ----------
    .. [#] C. J. Schrijver, K. L. Harvey, "The Photospheric Magnetic Flux
       Budget," Solar Physics, March 1994.
    """
    n_b = 5  # area bins
    delta_lnA = 0.5  # delta ln(A)
    bipole_max = 100.0  # orig. area of largest bipoles (deg^2)
    dcon = 2 * np.sinh(delta_lnA / 2)
    redu = np.exp(delta_lnA * np.arange(n_b))  # area reduction factors
    cum_redu = np.cumsum(redu)
    tot_redu = cum_redu[-1]  # sum of reduction factors
    widths = np.sqrt(bipole_max / redu)  # bipole separations (deg)
    b = 250.0 * (widths / 10.0) ** 2
    t_corr_min = 5  # first and last times (days)
    t_corr_max = 15  # for "correlated" emergence
    corr_prob = 1e-3  # probability for "correlation"
    n_lat = 16  # latitude bins
    n_lon = 36  # longitude bins
    rng = np.random.default_rng(random_seed)
    last_emergence = np.zeros((n_lon, n_lat, 2), int) + t_corr_max
    d_lat = lmax / n_lat
    d_lon = 360.0 / n_lon
    tmax = []
    lats = []
    lons = []
    bmax = []
    t = np.arange(t_start, t_sim)
    t = np.repeat(t, 2)
    cycle = t // (365 * pcyc)
    cycle[1::2] -= 1
    cycle_start = cycle * 365 * pcyc
    phase = (t - cycle_start) / (365 * (pcyc + pover))
    t = t[phase < 1]
    phase = phase[phase < 1]
    if not randomize:
        mu_lat = lmax + (lmin - lmax) * phase
        sd_lat = (lmax / 5.0) - 5 * phase
        lat1 = np.maximum(lmax * 0.9 - 1.2 * lmax * phase, 0.0) // d_lat
        lat2 = np.minimum(lmax + 15.0 - lmax * phase, lmax) // d_lat
    else:
        mu_lat = (lmax + lmin) / 2.0
        sd_lat = lmax - lmin
        lat1 = np.ones_like(phase) * lmin // d_lat
        lat2 = np.ones_like(phase) * lmax // d_lat
    lat1 = np.maximum(lat1, 0).astype(int)
    lat2 = np.minimum(lat2, n_lat - 1).astype(int)
    # Schrijver and Harvey (1994)
    uncorr_rate = 10 * act * dcon * np.sin(np.pi * phase) ** 2
    uncorr_rate, lat_bins = np.meshgrid(uncorr_rate, np.arange(n_lat))
    p = np.exp(-(((d_lat * (0.5 + lat_bins) - mu_lat) / sd_lat) ** 2))
    p[np.where((lat_bins < lat1) | (lat_bins > lat2))] = 0.0
    uncorr_rate *= p / (p.sum(axis=0) * n_lon * 2 * bipole_max)
    random_tries = rng.uniform(0.0, 1.0, uncorr_rate.shape)
    for i in range(len(phase)):
        if i == 0 or t[i] > t[i - 1]:
            last_emergence += 1
            # Initialize rate of emergence for largest regions
            corr_rate = (
                np.logical_and(
                    last_emergence > t_corr_min, last_emergence <= t_corr_max
                )
                * corr_prob
                / (t_corr_max - t_corr_min)
            )
        for hemisphere in [0, 1]:
            for lat_bin in range(lat1[i], lat2[i] + 1):
                lon_rate = uncorr_rate[lat_bin, i] + corr_rate[:, lat_bin, hemisphere]
                cum_rate = np.cumsum(lon_rate)
                tot_rate = cum_rate[-1]
                if random_tries[lat_bin, i] < tot_rate * tot_redu:
                    b_bin = np.digitize(random_tries[lat_bin, i], cum_redu * tot_rate)
                    tot_b = tot_rate * (cum_redu[b_bin] - redu[b_bin])
                    lon_bin = np.digitize(
                        random_tries[lat_bin, i], tot_b + cum_rate * redu[b_bin]
                    )
                    lon = d_lon * (rng.uniform() + lon_bin)
                    if lon > 180:
                        lon -= 360
                    lat = d_lat * (rng.uniform() + lat_bin)
                    tmax.append(t[i])
                    lats.append(lat * (1 - 2 * hemisphere) * np.pi / 180)
                    lons.append(lon * np.pi / 180)
                    bmax.append(b[b_bin])
                    if b_bin <= 1:
                        last_emergence[lon_bin, lat_bin, hemisphere] = 0
    tmax = np.array(tmax)
    lats = np.array(lats)
    lons = np.array(lons)
    bmax = np.array(bmax)
    return tmax, lats, lons, bmax


def inpaint_kepler(t, flux, imax=100, verbose=False):
    """Gaps filled with inpainting

    Parameters
    ----------
    t, y: irregularly sampled time series

    Returns
    -------
    inp_reg: inpainted regular sampled data
    """
    (bad,) = np.where(flux == 0.0)
    iflux = run_mca1d(flux, imax=imax, verbose=verbose)
    n = iflux.size
    inp_reg = np.empty((2, n))
    inp_reg[0, :] = t
    inp_reg[1, :] = flux
    inp_reg[1, bad] = iflux[bad]
    return inp_reg


def size_gap(in_data):
    """Search the size of the larger gap in a 1d signal

    Parameters
    ----------
    in_data: incomplete 1D data

    Returns
    -------
    max size of the gaps
    """
    (index,) = np.where(in_data != 0.0)
    gap_size = np.diff(index)
    return np.max(gap_size)


def run_mca1d(in_data, imax=100, verbose=False):
    """Process the inpainting of an incomplete 1d signal using Multiscale Discrete Cosine Transform

    Parameters
    ----------
    in_data: incomplete 1d signal

    Returns
    -------
    out_data: inpainted data
    """
    (ind,) = np.where(in_data != 0.0)
    (nind,) = np.where(in_data == 0.0)
    m = np.mean(in_data[ind])
    if verbose:
        print("mean = ", m)
        print("noise = ", np.std(np.fft.fft(in_data[ind]).real))
    in_data2 = in_data - m
    in_data2[nind] = 0.0
    mgap = size_gap(in_data2)
    # nscale = int(np.log(mgap) / np.log(2)) + 1.
    y = int(np.log(mgap * 8.0) / np.log(2)) + 1.0
    DCTBlockSize = int(2 ** y)
    if verbose:
        print(f"**** DCTBlockSize = {DCTBlockSize}")
        print(f"**** ITER = {imax}")
        out_data2 = cb_mca1d(
            in_data2, opt=f"-H -s0 -t3 -O -B {DCTBlockSize} " f"-L -v -i {imax} -D2"
        )
    else:
        out_data2 = cb_mca1d(
            in_data2, opt=f"-H -s0 -t3 -O -B {DCTBlockSize} " f"-L -i {imax} -D2"
        )
    out_data = out_data2 + m
    return out_data


def cb_mca1d(imag, opt):
    """Inpainting by decomposition of an image on multible bases

    Parameters
    ----------
    imag: 2D array
        image we want to decompose

    Returns
    -------
    result: image inpainted
    """
    PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
    noise = get_noise(imag)
    filename = "result"
    nameimag = f"tmp{np.random.randint(1e6)}.fits"
    fits.writeto(nameimag, imag)
    cmd = f"{PACKAGEDIR}/cb_mca1d {opt} -g {noise} {nameimag} {filename}"
    subprocess.call(cmd.split())
    result = fits.open(f"{filename}.fits")[0].data
    os.remove(nameimag)
    os.remove(f"{filename}.fits")
    os.remove(f"{filename}_cos.fits")
    os.remove(f"{filename}_resi.fits")
    return result


def zero_gaps(t, y, ts=None):
    if ts is None:
        ts = float(np.median(np.diff(t)))
    (gaps,) = np.where(np.diff(t) > 1.5 * ts)
    # t_gaps = []
    # y_gaps = []
    tnew = np.copy(t)
    ynew = np.copy(y)
    for g in gaps:
        t0, t1 = tnew[g : g + 2]
        tfill = np.arange(t0 + ts, t1, ts)
        tnew = np.append(tnew, tfill)
        ynew = np.append(ynew, np.zeros_like(tfill))
    sorted_args = tnew.argsort()
    tnew = tnew[sorted_args]
    ynew = ynew[sorted_args]
    return tnew, ynew


def fill_gaps(t, y, ts=None):
    """Linear interpolation to create a uniformly sampled signal

    Parameters
    ----------
    t: array-like
        signal timestamps
    y: array-like
        signal samples
    ts: float, optional
        sampling period; if timestamps are sorted, can be estimated if omitted

    Returns
    -------
    tnew: ndarray
        uniformly sampled time array
    ynew: ndarray
        interpolated signal samples
    """
    if ts is None:
        ts = float(np.median(np.diff(t)))
    tnew = np.arange(np.min(t), np.max(t), ts)
    ynew = interpolate.interp1d(t, y)(tnew)
    return tnew, ynew


def find_extrema(y, delta=0.0):
    """Finds local extrema

    Parameters
    ----------
    y: array-like
        signal array
    delta: float, optional
         minimum peak prominence

    Returns
    -------
    peaks: array-like
        maxima indices
    dips: array-like
        minima indices
    """
    maxima, _ = signal.find_peaks(y, prominence=delta)
    minima, _ = signal.find_peaks(-y, prominence=delta)
    return maxima, minima


def find_zero_crossings(y, height=None, delta=0.0):
    """Finds zero crossing indices

    Parameters
    ----------
    y: array-like
        signal
    height: float, optional
        maximum deviation from zero
    delta: float, optional
        prominence used in `scipy.signal.find_peaks` when `height` is specified

    Returns
    -------
    indzer: array-like
        zero-crossing indices
    """
    if height is None:
        (indzer,) = np.where(np.diff(np.signbit(y)))
    else:
        indzer, _ = signal.find_peaks(-np.abs(y), height=-height, prominence=delta)
    return indzer


def get_envelope(y, t=None, delta=0.0, nbsym=0):
    """Interpolates maxima and minima with cubic splines to derive upper and lower envelopes.

    Parameters
    ----------
    y: array-like
        signal
    t: array-like, optional
        signal timestamps
    delta: float, optional
        prominence to use in `find_extrema`
    nbsym: int, optional
        number of extrema to repeat on either side of the signal

    Returns
    -------
    upper: array-like
        upper envelope
    lower: array-like
        lower envelope
    """
    if t is None:
        t = np.arange(len(y))

    peaks, dips = find_extrema(y, delta)
    if nbsym == 0:
        peaks = np.r_[0, peaks, len(y) - 1]
        tmax = t[peaks]
        ymax = y[peaks]
        dips = np.r_[0, peaks, len(y) - 1]
        tmin = t[dips]
        ymin = y[dips]
    else:
        lpeaks = peaks[:nbsym][::-1]
        rpeaks = peaks[-nbsym:][::-1]
        loff = 2 * t[0] - t[lpeaks]
        roff = 2 * t[-1] - t[rpeaks]
        tmax = np.r_[loff, t[peaks], roff]
        ymax = np.r_[y[lpeaks], y[peaks], y[rpeaks]]
        ldips = dips[:nbsym][::-1]
        rdips = dips[-nbsym:][::-1]
        loff = 2 * t[0] - t[ldips]
        roff = 2 * t[-1] - t[rdips]
        tmin = np.r_[loff, t[dips], roff]
        ymin = np.r_[y[ldips], y[dips], y[rdips]]

    tck = interpolate.splrep(tmax, ymax)
    upper = interpolate.splev(t, tck)
    tck = interpolate.splrep(tmin, ymin)
    lower = interpolate.splev(t, tck)
    return upper, lower


def smooth(y, kernel):
    """Wrap to numpy.convolve

    Parameters
    ----------
    y: array-like
        input noisy signal
    kernel: array-like
        FIR filter to smooth the signal
    Returns
    -------
    yf: array-like
        Smoothed signal
    """
    w = kernel.shape[0]
    s = np.r_[y[w - 1 : 0 : -1], y, y[-2 : -w - 1 : -1]]
    sf = np.convolve(s, kernel, mode="valid")
    yf = sf[w // 2 : -w // 2 + 1]
    return yf


def get_noise(y, sigma=3.0, niter=3):
    """Finds the standard deviation of a white gaussian noise in the data

    Parameters
    ----------
    y: array-like
        signal array
    sigma: float, optional (default=3.0)
        sigma_clip value
    niter: int, optional (default=3)
        number of iterations for k-sigma clipping

    Returns
    -------
    noise: float
        estimate of standard deviation of the noise
    """
    resid = y - signal.medfilt(y, 3)
    sd = np.std(resid)
    index = np.arange(resid.size)
    for i in range(niter):
        mu = np.mean(resid[index])
        sd = np.std(resid[index])
        (index,) = np.where(np.abs(resid - mu) < sigma * sd)
    noise = sd / 0.893421
    return noise
