import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def supermongo_mplstyle():
    ## supermongo sans-serif plot style
    plt.rc("lines", linewidth=1.0, linestyle="-", color="black")
    plt.rc("font", family="sans-serif", weight="normal", size=12.0)
    plt.rc("text", color="black", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{cmbright}")
    plt.rc(
        "axes",
        edgecolor="black",
        facecolor="white",
        linewidth=1.0,
        grid=False,
        titlesize="x-large",
        labelsize="x-large",
        labelweight="normal",
        labelcolor="black",
    )
    plt.rc("axes.formatter", limits=(-4, 4))
    plt.rc(("xtick", "ytick"), labelsize="x-large", direction="in")
    plt.rc("xtick", top=True)
    plt.rc("ytick", right=True)
    plt.rc(("xtick.major", "ytick.major"), size=7, pad=6, width=1.0)
    plt.rc(("xtick.minor", "ytick.minor"), size=4, pad=6, width=1.0, visible=True)
    plt.rc("legend", numpoints=1, fontsize="x-large", shadow=False, frameon=False)


def plot_sun(x, y, ax=plt):
    ax.plot(x, y, "ko", ms=10)
    ax.plot(x, y, "yo", ms=9)
    ax.plot(x, y, "k.", ms=2)


def plot_mcmc(
    samples, labels=None, priors=None, ptrue=None, precision=None, nbins=30, s=1.0
):
    """Plots a Giant Triangle Confusogram

    Parameters
    ----------
    samples: 2-D array, shape (N, ndim)
        Samples from ndim variables to be plotted in the GTC
    labels: list of strings, optional
        List of names for each variable (size ndim)
    priors: list of callables, optional
        List of prior functions for the variables distributions (size ndim)
    ptrue: list of floats, optional
        List of true estimates for each parameter
    precision: list of ints, optional
        List of decimal places to write down for each parameter. Defaults to 2
    nbins: int, optional
        Number of bins to be used in 1D and 2D histograms. Defaults to 30
    s: float, optional
        Standard deviation of Gaussian filter applied to smooth histograms. Defaults to 1.0
    """
    p = map(
        lambda v: (v[1], v[1] - v[0], v[2] - v[1]),
        zip(*np.percentile(samples, [16, 50, 84], axis=0)),
    )
    p = list(p)
    ndim = samples.shape[-1]
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # TODO: pyplot style context
    grid = plt.GridSpec(ndim, ndim, wspace=0.0, hspace=0.0)
    handles = []

    if precision is None:
        precision = 2 * np.ones(ndim, dtype=int)
    if ptrue is None:
        ptrue = np.array([None for _ in range(ndim)])

    # PLOT 1D
    for i in range(ndim):
        ax = fig.add_subplot(grid[i, i])
        H, edges = np.histogram(samples[:, i], bins=nbins, density=True)
        centers = (edges[1:] + edges[:-1]) / 2
        data = ndimage.gaussian_filter1d((centers, H), sigma=s)
        data[1] /= data[1].sum()
        (l1,) = ax.plot(data[0], data[1], "b-", lw=1, label="posterior")
        if priors is not None:
            pr = priors[i](centers)
            pr /= pr.sum()
            (l2,) = ax.plot(centers, pr, "k-", lw=1, label="prior")
        l3 = ax.axvline(p[i][0], color="k", ls="--", label="median")
        mask = np.logical_and(
            centers - p[i][0] <= p[i][2], p[i][0] - centers <= p[i][1]
        )
        ax.fill_between(
            centers[mask], np.zeros(mask.sum()), data[1][mask], color="b", alpha=0.3
        )
        if ptrue[i] is not None:
            l4 = ax.axvline(ptrue[i], color="gray", lw=1.5, label="true")
        if i < ndim - 1:
            ax.set_xticks([])
        else:
            ax.tick_params(rotation=45)
        ax.set_yticks([])
        ax.get_xaxis().set_major_locator(plt.MaxNLocator(3))
        ax.set_ylim(0)
        if labels is not None:
            ax.set_title(
                "{0} = {1:.{4}f}$^{{+{2:.{4}f}}}_{{-{3:.{4}f}}}$".format(
                    labels[i], p[i][0], p[i][2], p[i][1], precision[i]
                )
            )

    handles.append(l1)
    try:
        handles.append(l2)
    except UnboundLocalError:
        pass
    try:
        handles.append(l3)
    except UnboundLocalError:
        pass
    try:
        handles.append(l4)
    except UnboundLocalError:
        pass

    # PLOT 2D
    nbins_flat = np.linspace(0, nbins ** 2, nbins ** 2)
    for i in range(ndim):
        for j in range(i):
            ax = fig.add_subplot(grid[i, j])
            H, xi, yi = np.histogram2d(samples[:, j], samples[:, i], bins=nbins)
            extents = [xi[0], xi[-1], yi[0], yi[-1]]
            H /= H.sum()
            H_order = np.sort(H.flat)
            H_cumul = np.cumsum(H_order)
            tmp = np.interp([0.0455, 0.3173, 1.0], H_cumul, nbins_flat)
            chainlevels = np.interp(tmp, nbins_flat, H_order)
            data = ndimage.gaussian_filter(H.T, sigma=s)
            xbins = (xi[1:] + xi[:-1]) / 2
            ybins = (yi[1:] + yi[:-1]) / 2
            ax.contourf(
                xbins,
                ybins,
                data,
                levels=chainlevels,
                colors=["#1f77b4", "#52aae7", "#85ddff"],
                alpha=0.3,
            )
            ax.contour(data, chainlevels, extent=extents, colors="b")
            ax.get_xaxis().set_major_locator(plt.MaxNLocator(3))
            ax.get_yaxis().set_major_locator(plt.MaxNLocator(3))
            if ptrue[i] is not None:
                ax.axhline(ptrue[i], color="gray", lw=1.5)
            if ptrue[j] is not None:
                ax.axvline(ptrue[j], color="gray", lw=1.5)
            if i < ndim - 1:
                ax.set_xticks([])
            else:
                ax.tick_params(rotation=45)
            if j > 0:
                ax.set_yticks([])
            else:
                ax.tick_params(rotation=45)
    fig.legend(handles=handles)
    return fig
