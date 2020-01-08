import matplotlib.pyplot as plt


def supermongo_mplstyle():
    ## supermongo sans-serif plot style
    plt.rc('lines', linewidth=1.0, linestyle='-', color='black')
    plt.rc('font', family='sans-serif', weight='normal', size=12.0)
    plt.rc('text', color='black', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rc('axes', edgecolor='black', facecolor='white', linewidth=1.0, grid=False,
           titlesize='x-large', labelsize='x-large', labelweight='normal', labelcolor='black')
    plt.rc('axes.formatter', limits=(-4, 4))
    plt.rc(('xtick', 'ytick'), labelsize='x-large', direction='in')
    plt.rc('xtick', top=True)
    plt.rc('ytick', right=True)
    plt.rc(('xtick.major', 'ytick.major'), size=7, pad=6, width=1.0)
    plt.rc(('xtick.minor', 'ytick.minor'), size=4, pad=6, width=1.0, visible=True)
    plt.rc('legend', numpoints=1, fontsize='x-large', shadow=False, frameon=False)


def plot_sun(x, y, ax=plt):
    ax.plot(x, y, 'ko', ms=10)
    ax.plot(x, y, 'yo', ms=9)
    ax.plot(x, y, 'k.', ms=2)
