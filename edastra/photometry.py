import numpy as np

def BPRP_to_BV_Evans2018(bprp):
    # -0.5 < bprp < 2.75
    # DANGER: -0.27 <= bppr <= 0.23
    bprp = np.atleast_1d(bprp)
    # sigma = 0.045858
    gv = np.poly1d([-0.1732, -0.006860, -0.01760])(bprp)
    # sigma = 0.06285
    bv = np.array([np.roots([-0.001768, -0.2297, -0.02385, -0.02907-x])[2] for x in gv])
    return bv


def BPRP_to_GV_Evans2018(bprp):
    # -0.5 < bprp < 2.75
    bprp = np.atleast_1d(bprp)
    # sigma = 0.045858
    gv = np.poly1d([-0.1732, -0.006860, -0.01760])(bprp)
    return gv