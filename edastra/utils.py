import subprocess
import os

import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel
from scipy import interpolate, signal


def prot_acf(t, y, max_per=None, s=0, fill=True, criterium='height', **kwargs):
    if max_per is not None:
        max_per = float(max_per)
    lags, ryy = acf(y, t, s=s, fill=fill, maxlag=max_per)
    peaks, properties = signal.find_peaks(ryy, **kwargs)
    if criterium == 'height':
        metric = ryy[peaks]
    elif criterium == 'prominence':
        metric = properties['prominences']
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


def inpaint_kepler(t, flux, imax=100, verbose=False):
    """Gaps filled with inpainting
    
    Parameters
    ----------
    t, y: irregularly sampled time series
    
    Returns
    -------
    inp_reg: inpainted regular sampled data
    """
    bad, = np.where(flux == 0.0)
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
    index, = np.where(in_data != 0.0)
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
    ind, = np.where(in_data != 0.0)
    nind, = np.where(in_data == 0.0)
    m = np.mean(in_data[ind])
    if verbose:
        print('mean = ', m)
        print('noise = ', np.std(np.fft.fft(in_data[ind]).real))
    in_data2 = in_data - m
    in_data2[nind] = 0.0
    mgap = size_gap(in_data2)
    nscale = int(np.log(mgap) / np.log(2)) + 1.
    y = int(np.log(mgap * 8.) / np.log(2)) + 1.
    DCTBlockSize = int(2 ** y)
    if verbose:
        print('**** DCTBlockSize = ', DCTBlockSize)
        print('**** ITER = ', imax)
        out_data2 = cb_mca1d(in_data2, opt=f'-H -s0 -t3 -O -B {DCTBlockSize} '
                                           f'-L -v -i {imax} -D2')
    else:
        out_data2 = cb_mca1d(in_data2, opt=f'-H -s0 -t3 -O -B {DCTBlockSize} '
                                           f'-L -i {imax} -D2')
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
    filename = 'result'
    nameimag = f'tmp{np.random.randint(1e6)}.fits'
    fits.writeto(nameimag, imag)
    cmd = f'{EXECDIR}/cb_mca1d {opt} -g {noise} {nameimag} {filename}'
    subprocess.call(cmd.split())
    result = fits.open(f'{filename}.fits')[0].data
    os.remove(nameimag)
    os.remove(f'{filename}.fits')
    os.remove(f'{filename}_cos.fits')
    os.remove(f'{filename}_resi.fits')
    return result


def zero_gaps(t, y, ts=None):
    if ts is None:
        ts = float(np.median(np.diff(t)))
    gaps, = np.where(np.diff(t) > 1.5 * ts)
    t_gaps = []
    y_gaps = []
    tnew = np.copy(t)
    ynew = np.copy(y)
    for g in gaps:
        t0, t1 = tnew[g:g+2]
        tfill = np.arange(t0+ts, t1, ts)
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
    tnew: array-like
        uniformly sampled time array
    ynew: array-like
        interpolated signal samples
    """
    if ts is None:
        ts = float(np.median(np.diff(t)))
    tnew = np.arange(np.min(t), np.max(t), ts)
    ynew = interpolate.interp1d(t, y)(tnew)
    return tnew, ynew


def find_extrema(y, delta=0.):
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


def find_zero_crossings(y, height=None, delta=0.):
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
        indzer, = np.where(np.diff(np.signbit(y)))
    else:
        indzer, _ = signal.find_peaks(-np.abs(y), height=-height, prominence=delta)
    return indzer


def get_envelope(y, t=None, delta=0., nbsym=0):
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
    s = np.r_[y[w-1:0:-1], y, y[-2:-w-1:-1]]
    sf = np.convolve(s, kernel, mode='valid')
    yf = sf[w // 2 - 1:-w // 2]
    return yf


def get_noise(y, sigma=3., niter=3):
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
        index, = np.where(np.abs(resid - mu) < sigma * sd)
    noise = sd / .893421
    return noise
