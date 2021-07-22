import os

from astropy import table
import numpy as np
from scipy import interpolate


def mass_Noyes1984(bv, bv_err=None):
    """Noyes et al. 1984, ApJ, 279, 763"""
    # 0.4 < bv < 1.4
    bv = np.atleast_1d(bv)
    f = np.poly1d([-0.42, 0.28])
    fdot = np.polyder(f)
    logtol = 0.01
    logmass = f(bv)
    mass = 10 ** logmass
    if bv_err is not None:
        bv_err = np.atleast_1d(bv_err)
        bv_var = bv_err ** 2
        logmass_var = bv_var * fdot(bv) ** 2 + logtol ** 2
        mass_var = logmass_var * (np.log(10) * mass) ** 2
        mass_err = np.sqrt(mass_var)
        return mass, mass_err
    return mass


def teff_Noyes1984(bv, bv_err=None):
    """Noyes et al. 1984, ApJ, 279, 763"""
    # 0.4 < bv < 1.4
    bv = np.atleast_1d(bv)
    f = np.poly1d([-0.234, 3.908])
    fdot = np.polyder(f)
    logtol = 0.002
    logteff = f(bv)
    teff = 10 ** logteff
    if bv_err is not None:
        bv_err = np.atleast_1d(bv_err)
        bv_var = bv_err ** 2
        logteff_var = bv_var * fdot(bv) ** 2 + logtol ** 2
        teff_var = logteff_var * (np.log(10) * teff) ** 2
        teff_err = np.sqrt(teff_var)
        return teff, teff_err
    return teff


def bv_Noyes1984(teff, teff_err=None):
    """Noyes et al. 1984, ApJ, 279, 763"""
    # 0.4 < bv < 1.4
    teff = np.atleast_1d(teff)
    logteff = np.log10(teff)
    f = np.poly1d([-1 / 0.234, 3.908 / 0.234])
    fdot = np.polyder(f)
    bv = f(logteff)
    if teff_err is not None:
        teff_err = np.atleast_1d(teff_err)
        teff_var = teff_err ** 2
        logteff_var = teff_var / (np.log(10) * teff) ** 2
        bv_var = logteff_var * fdot(logteff) ** 2
        bv_err = np.sqrt(bv_var)
        return bv, bv_err
    return bv


def tau_Noyes1984(bv, bv_err=None):
    """Noyes et al. 1984, ApJ, 279, 763"""
    bv = np.atleast_1d(bv)
    x = 1 - bv
    mask = x < 0
    f = np.poly1d([-5.323, 0.025, -0.166, 1.362])
    fdot = np.polyder(f)
    g = np.poly1d([-0.14, 1.362])
    gdot = np.polyder(g)
    logtau = f(x)
    logtau[mask] = g(x[mask])
    tau = 10 ** logtau
    if bv_err is not None:
        bv_err = np.atleast_1d(bv_err)
        bv_var = bv_err ** 2
        x_var = bv_var
        logtau_var = x_var * fdot(x) ** 2
        logtau_var[mask] = (x_var * gdot(x) ** 2)[mask]
        tau_var = logtau_var * (np.log(10) * tau) ** 2
        tau_err = np.sqrt(tau_var)
        return tau, tau_err
    return tau


def Rossby_Noyes1984(prot, bv, prot_err=None, bv_err=None):
    """Noyes et al. 1984, ApJ, 279, 763"""
    if bv_err is not None:
        tau, tau_err = tau_Noyes1984(bv, bv_err)
    else:
        tau = tau_Noyes1984(bv)
    prot = np.atleast_1d(prot)
    ro = prot / tau
    if prot_err is not None and bv_err is not None:
        prot_err = np.atleast_1d(prot_err)
        ro_err = np.hypot(prot_err / tau, ro * tau_err / tau)
        return ro, ro_err
    return ro


def tau_Saar1999(bv, bv_err=None):
    """Saar & Brandenburg 1999, ApJ, 524, 295"""
    bv = np.atleast_1d(bv)
    mask = bv >= 1
    f = np.poly1d([-3.1466, 12.540, -20.063, 15.382, -3.3300])
    fdot = np.polyder(f)
    logtau = f(bv)
    logtau[mask] = np.log10(25.0)
    tau = 10 ** logtau
    if bv_err is not None:
        bv_err = np.atleast_1d(bv_err)
        bv_var = bv_err ** 2
        logtau_var = bv_var * fdot(bv) ** 2
        logtau_var[mask] = 0.0
        tau_var = logtau_var * (np.log(10) * tau) ** 2
        tau_err = np.sqrt(tau_var)
        return tau, tau_err
    return tau


def Rossby_Saar1999(prot, bv, prot_err=None, bv_err=None):
    """Saar & Brandenburg 1999, ApJ, 524, 295"""
    if bv_err is not None:
        tau, tau_err = tau_Saar1999(bv, bv_err)
    else:
        tau = tau_Saar1999(bv)
    prot = np.atleast_1d(prot)
    ro = prot / (4 * np.pi * tau)
    if prot_err is not None and bv_err is not None:
        prot_err = np.atleast_1d(prot_err)
        ro_err = np.hypot(prot_err / tau, ro * tau_err / tau) / (4 * np.pi)
        return ro, ro_err
    return ro


def Gyro_Barneslike(
    n,
    a,
    b,
    c,
    bv,
    n_err=0,
    a_err=0,
    b_err=0,
    c_err=0,
    bv_err=None,
    age=None,
    prot=None,
    age_err=None,
    prot_err=None,
    diff_rot=False,
):
    bv = np.atleast_1d(bv)
    if age is not None:
        logprot = n * np.log(age) + b * np.log(bv - c) + np.log(a)
        prot = np.exp(logprot)
        if bv_err is not None and age_err is not None:
            bv_err = np.atleast_1d(bv_err)
            age_err = np.atleast_1d(age_err)
            logage_var = (age_err / age) ** 2
            bvc_err = np.hypot(bv_err, c_err)
            logbvc_var = (bvc_err / (bv - c)) ** 2
            n_var = n_err ** 2
            b_var = b_err ** 2
            loga_var = (a_err / a) ** 2
            logprot_var = (
                n_var * np.log(age) ** 2
                + logage_var * n ** 2
                + b_var * np.log(bv - c) ** 2
                + logbvc_var * b ** 2
                + loga_var
            )
            prot_var = logprot_var * prot ** 2
            prot_err = np.sqrt(prot_var)
            return prot, prot_err
        return prot
    elif prot is not None:
        logage = (np.log(prot) - b * np.log(bv - c) - np.log(a)) / n
        age = np.exp(logage)
        if bv_err is not None and prot_err is not None:
            bv_err = np.atleast_1d(bv_err)
            prot_err = np.atleast_1d(prot_err)
            logprot_var = (prot_err / prot) ** 2
            if diff_rot:
                logprot_var += (10 ** -1.85 * prot ** 0.3) ** 2
            bvc_err = np.hypot(bv_err, c_err)
            logbvc_var = (bvc_err / (bv - c)) ** 2
            n_var = n_err ** 2
            b_var = b_err ** 2
            loga_var = (a_err / a) ** 2
            nlogage_var = (
                logprot_var
                + loga_var
                + b_var * np.log(bv - c) ** 2
                + logbvc_var * b ** 2
            )
            nlogage_err = np.sqrt(nlogage_var)
            logage_err = np.hypot(nlogage_err / n, logage * n_err / n)
            age_var = (logage_err * age) ** 2
            age_err = np.sqrt(age_var)
            return age, age_err
        return age
    else:
        raise ValueError("One of (age, prot) must be given")


def Gyro_Barnes2007(bv, **kwargs):
    """Barnes 2007, ApJ, 669, 1167"""
    return Gyro_Barneslike(
        n=0.5189,
        a=0.7725,
        b=0.601,
        c=0.40,
        n_err=0.007,
        a_err=0.011,
        b_err=0.024,
        c_err=0.0,
        bv=bv,
        **kwargs,
    )


def Gyro_Mamajek2008(bv, **kwargs):
    """Mamajek & Hillenbrand 2008, ApJ, 687, 1264"""
    return Gyro_Barneslike(
        n=0.566,
        a=0.407,
        b=0.325,
        c=0.495,
        n_err=0.008,
        a_err=0.021,
        b_err=0.024,
        c_err=0.010,
        bv=bv,
        **kwargs,
    )


def Gyro_Angus2015(bv, **kwargs):
    """Angus et al. 2015, MNRAS, 450, 1787"""
    return Gyro_Barneslike(n=0.55, a=0.4, b=0.31, c=0.45, bv=bv, **kwargs)


def tau_BarnesKim2010(mass=None, teff=None, mass_err=None):
    fname = os.path.join(os.path.dirname(__file__), "data/barnes_kim_2010.csv")
    tbl = table.Table.read(fname)
    if mass is not None:
        f = interpolate.interp1d(tbl["M"], tbl["globalToT"], fill_value="extrapolate")
        tau = f(mass)
        if mass_err is not None:
            mass_err = np.atleast_1d(mass_err)
            dxs = (f.x[1:] + f.x[:-1]) / 2
            dys = np.diff(f.y) / np.diff(f.x)
            df = interpolate.interp1d(dxs, dys, "nearest")
            tau_err = np.abs(df(mass) * mass_err)
            return tau, tau_err
        return tau
    elif teff is not None:
        f = interpolate.interp1d(
            tbl["logT"], tbl["globalToT"], fill_value="extrapolate"
        )
        logteff = np.log10(teff)
        tau = f(logteff)
        return tau
    else:
        raise ValueError("One of (mass, teff) must be given")


def Gyro_Barnes2010(tau, prot, P0=1.1, prot_err=None, tau_err=None, initial_err=False):
    k_c = 0.646
    k_i = 452
    age = (tau / k_c) * np.log(prot / P0) + (k_i / (2 * tau)) * (prot ** 2 - P0 ** 2)
    if tau_err is not None and prot_err is not None:
        prot_err = np.atleast_1d(prot_err)
        tau_err = np.atleast_1d(tau_err)
        prot_var = prot_err ** 2
        tau_var = tau_err ** 2
        age_var = (
            prot_var * ((tau / k_c) / prot + (k_i / tau) * prot) ** 2
            + tau_var
            * (np.log(prot / P0) / k_c - (prot ** 2 - P0 ** 2) * (k_i / 2) / tau ** 2)
            ** 2
        )
        if initial_err:
            age_var += (Gyro_Barnes2010(tau, prot=3.4, P0=0.12) / 2) ** 2
        age_err = np.sqrt(age_var)
        return age, age_err
    return age
