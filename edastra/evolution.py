import numpy as np


def tau_Noyes1984(bv):
    bv = np.atleast_1d(bv)
    x = 1 - bv
    logtau = np.poly1d([-5.323, 0.025, -0.166, 1.362])(x)
    logtau[x < 0] = 1.362 - 0.14 * x[x < 0]
    return 10 ** logtau


def Rossby_Noyes1984(prot, bv):
    return prot / tau_Noyes1984(bv)


def tau_Saar1999(bv):
    """Saar & Brandenburg 1999, """
    bv = np.atleast_1d(bv)
    logtau = np.poly1d([-3.1466, 12.540, -20.063, 15.382, -3.3300])(bv)
    logtau[bv >= 1] = np.log10(25.)
    return 10 ** logtau


def Rossby_Saar1999(prot, bv):
    return prot / (4 * np.pi * tau_Saar1999(bv))


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
    """Barnes 2007, """
    return Gyro_Barneslike(n=.5189, a=.7725, b=.601, c=.40, 
                           bv=bv, age=age, prot=prot)

    
def Gyro_Mamajek2008(bv, age=None, prot=None):
    """Mamajek & Hillenbrand 2008, """
    return Gyro_Barneslike(n=.566, a=.407, b=.325, c=.495,
                           bv=bv, age=age, prot=prot)


def Gyro_Angus2015(bv, age=None, prot=None):
    return Gyro_Barneslike(n=.55, a=.4, b=.31, c=.45,
                           bv=bv, age=age, prot=prot)

