import numpy as np


def flux_to_mag(flux, flux0, mag0=0., flux_err=None):
    flux = np.atleast_1d(flux)
    mag = -2.5 * np.log10(flux / flux0) + mag0
    if flux_err is not None:
        flux_err = np.atleast_1d(flux_err)
        mag_err = flux_err * 2.5 / (np.log(10) * flux)
        return mag, mag_err
    return mag


def BPRP_to_BV_Evans2018(bprp, bprp_err=None):
    """Evans et al. 2018, A&A, 616, A4"""
    # -0.5 < bprp < 2.75
    # DANGER: -0.27 <= bppr <= 0.23
    bprp = np.atleast_1d(bprp)
    f = np.poly1d([-0.1732, -0.006860, -0.01760])
    fdot = np.polyder(f)
    sigma = 0.045858
    gv = f(bprp)
    if bprp_err is not None:
        bprp_err = np.atleast_1d(bprp_err)
        bprp_var = bprp_err ** 2
        gv_var = bprp_var * fdot(bprp) ** 2 + sigma ** 2
        gv_err = np.sqrt(gv_var)
    # sigma = 0.06285
    # TODO
    bv = np.array([np.roots([-0.001768, -0.2297, -0.02385, -0.02907-x])[2] for x in gv])
    return bv


def BPRP_to_GV_Evans2018(bprp, bprp_err=None):
    """Evans et al. 2018, A&A, 616, A4"""
    # -0.5 < bprp < 2.75
    bprp = np.atleast_1d(bprp)
    f = np.poly1d([-0.1732, -0.006860, -0.01760])
    fdot = np.polyder(f)
    sigma = 0.045858
    gv = f(bprp)
    if bprp_err is not None:
        bprp_err = np.atleast_1d(bprp_err)
        gv_err = np.hypot(bprp_err * fdot(bprp), sigma)
        return gv, gv_err
    return gv


def BV_to_GV_Evans2018(bv, bv_err=None):
    """Evans et al. 2018, A&A, 616, A4"""
    bv = np.atleast_1d(bv)
    f = np.poly1d([-0.001768, -0.2297, -0.02385, -0.02907])
    fdot = np.polyder(f)
    sigma = 0.06285
    gv = f(bv)
    if bv_err is not None:
        bv_err = np.atleast_1d(bv_err)
        gv_err = np.hypot(bv_err * fdot(bv), sigma)
        return gv, gv_err
    return gv
