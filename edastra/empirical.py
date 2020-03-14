import numpy as np


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
        teff_var = logteff_var  * (np.log(10) * teff) ** 2
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
    logtau[mask] = np.log10(25.)
    tau = 10 ** logtau
    if bv_err is not None:
        bv_err = np.atleast_1d(bv_err)
        bv_var = bv_err ** 2
        logtau_var = bv_var * fdot(bv) ** 2
        logtau_var[mask] = 0.
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


def Gyro_Barneslike(n, a, b, c, bv, age=None, prot=None):
    bv = np.atleast_1d(bv).reshape(-1, 1)
    if age is not None:
        logprot = n * np.log(age) + b * np.log(bv - c) + np.log(a)
        prot = np.exp(logprot)
        return prot
    elif prot is not None:
        logage = (np.log(prot) - b * np.log(bv - c) - np.log(a)) / n
        age = np.exp(logage)
        return age
    else:
        raise ValueError("One of (age, prot) must be given")


def Gyro_Barnes2007(bv, age=None, prot=None):
    """Barnes 2007, ApJ, 669, 1167"""
    return Gyro_Barneslike(n=.5189, a=.7725, b=.601, c=.40, 
                           bv=bv, age=age, prot=prot)

    
def Gyro_Mamajek2008(bv, age=None, prot=None):
    """Mamajek & Hillenbrand 2008, ApJ, 687, 1264"""
    return Gyro_Barneslike(n=.566, a=.407, b=.325, c=.495,
                           bv=bv, age=age, prot=prot)


def Gyro_Angus2015(bv, age=None, prot=None):
    """Angus et al. 2015, MNRAS, 450, 1787"""
    return Gyro_Barneslike(n=.55, a=.4, b=.31, c=.45,
                           bv=bv, age=age, prot=prot)

