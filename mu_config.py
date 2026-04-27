import pandas as pd
import matplotlib.pyplot as plt


summer_months = [5, 6, 7, 8, 9]
summer_months_s = [11, 12, 1, 2, 3]


scat_c, line_c = '#305f72', '#d6a354'
depth_color = '#595959'


plt.rcParams.update({
    'font.size': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,

    'axes.linewidth': 0.5,
    'axes.labelsize': 9,
    'axes.titlesize': 10,

    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'grid.linewidth': 0.4,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'axes.grid': True,

    'figure.figsize': (8, 4),
    'figure.dpi': 100,

    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.transparent': False,
})