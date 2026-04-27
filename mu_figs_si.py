import copy
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patheffects import Normal
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from brokenaxes import brokenaxes
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.tri as tri
import chardet
from statsmodels.genmod.families import Family, Gaussian, Poisson, Binomial, Gamma, NegativeBinomial
from statsmodels.genmod.families.links import identity, log, logit, inverse_power
from scipy.stats import kruskal, mannwhitneyu, gamma
from itertools import combinations
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from pygam import LinearGAM, s, te
import pymc as pm
import arviz as az
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import warnings
import seaborn as sns
from sympy.abc import alpha
warnings.filterwarnings('ignore')


from mu_utils import *
from mu_config import *



def fs_lake_data_map():
    data_df = get_lake_wq()
    meta_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_meta.csv'), header=0, index_col=0)
    for c in meta_df.columns:
        if c not in data_df.columns:
            data_df[c] = data_df['Unique Lake'].map(meta_df[c])
    data_df = data_df.dropna(subset=['Latitude', 'Longitude', 'Chla (µg/L)', 'sampledate'])
    data_df = filter_summer_data(data_df, t_lat=10)
    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)
    mean_df = mean_df[mean_df['Chla (µg/L)'] >= 1]

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    m = Basemap(projection='cyl', resolution='i', ax=ax)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='lightgray', lake_color='lightgray', alpha=.5)
    m.drawmapboundary(fill_color='none', linewidth=0.5)
    parallels = np.arange(-80, 81, 20)
    meridians = np.arange(-180, 181, 60)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')

    x, y = m(mean_df['Longitude'].values, mean_df['Latitude'].values)
    values = np.log10(mean_df['Chla (µg/L)'].values)
    norm = mpl.colors.Normalize(vmin=np.percentile(values, 1), vmax=np.percentile(values, 99), clip=True)
    s = m.scatter(
        x, y,
        c=np.log10(mean_df['Chla (µg/L)'].values),
        cmap='viridis',
        norm=norm,
        marker='o',
        s=5,
        edgecolor='none',
        linewidths=0,
        alpha=1,
    )
    cb = m.colorbar(s, location='right', size="2%",pad='2%')
    cb.set_label("Mean growing-season Chla (µg/L)", fontsize=8)
    ticks = np.power(10, np.linspace(np.percentile(values, 0), np.percentile(values, 99), 7))
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels([str(int(t)) for t in ticks])
    cb.ax.tick_params(labelsize=8, width=.5, length=3)
    cb.outline.set_linewidth(.5)
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig('Figs/FS Lake water quality map.png', dpi=300)


def fs_lake_wq_site_dist_regression():
    used_data_df = filter_lakes(n_year=1, t_depth=100, if_mean=False, if_filter=False)
    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['depth (m)'] = used_data_df['depth']
    used_data_df['LN (µg P/L)'] = used_data_df['ln']

    cs = ['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)', 'TN:TP']
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))
    for i, c in enumerate(cs):
        plt.sca(axes[0, i])
        plt.hist(np.log10(used_data_df[c]), bins=25, density=True, color=scat_c, alpha=1, edgecolor=None)
        kde = stats.gaussian_kde(np.log10(used_data_df[c]))
        x_vals = np.linspace(min(np.log10(used_data_df[c])), max(np.log10(used_data_df[c])), 500)
        plt.plot(x_vals, kde(x_vals), color=line_c, linewidth=2)
        for spine in axes[0, i].spines.values():
            spine.set_linewidth(.5)
        axes[0, i].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel('Growing-season ' + c.replace('µg/L', 'log µg/L').replace('TN:TP', 'log TN:TP'), fontsize=8)
        plt.ylabel('Density', fontsize=8)

    for i_r, c_x in {1: 'depth (m)', 2: 'Absolute latitude'}.items():
        for i_c, c_y in enumerate(cs):
            plt.sca(axes[i_r, i_c])
            plt.scatter(used_data_df[c_x], np.log10(used_data_df[c_y]), alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
            lowess_result = sm.nonparametric.lowess(np.log10(used_data_df[c_y]), used_data_df[c_x], frac=.8)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2)
            for spine in axes[i_r, i_c].spines.values():
                spine.set_linewidth(.5)
            axes[i_r, i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
            plt.grid(lw=.5, alpha=0.4)
            plt.xlabel(c_x, fontsize=8)
            plt.ylabel('Growing-season ' + c_y.replace('µg/L', 'log µg/L').replace('TN:TP', 'log TN:TP'), fontsize=8)

    cs = ['TP (µg/L)', 'TN (µg/L)', 'depth (m)', 'Absolute latitude']
    c_y = 'Chla (µg/L)'
    for i, c_x in enumerate(cs):
        plt.sca(axes[3, i])
        plt.scatter(used_data_df[c_x], used_data_df[c_y], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(used_data_df[c_y], used_data_df[c_x], frac=.8)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2)
        for spine in axes[3, i].spines.values():
            spine.set_linewidth(.5)
        axes[3, i].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel('Growing-season ' + c_x.replace('µg/L', 'µg/L').replace('TN:TP', 'TN:TP'), fontsize=8)
        plt.ylabel('Growing-season ' + c_y.replace('µg/L', 'µg/L').replace('TN:TP', 'TN:TP'), fontsize=8)

    plt.tight_layout()
    plt.savefig('Figs/FS WQ along depth and lat.png', dpi=300)


def fs_lake_wq_dist_and_regression():
    used_data_df = filter_lakes(n_year=1, t_depth=100, if_mean=False, if_filter=False)
    print('filter n lake', len(used_data_df))
    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Depth (m)'] = used_data_df['depth']
    used_data_df['LN (µg P/L)'] = used_data_df['ln']
    used_data_df['Growing-season\nTP (µg/L)'] = used_data_df['TP (µg/L)']
    used_data_df['Growing-season\nTN (µg/L)'] = used_data_df['TN (µg/L)']
    used_data_df['Growing-season\nChla (µg/L)'] = used_data_df['Chla (µg/L)']
    print(len(used_data_df))

    def draw_lowess(x, y, frac=0.7, color=line_c, lw=1.5, **kwargs):
        smoothed = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color=color, lw=lw)

    cols = ['Growing-season\nTP (µg/L)', 'Growing-season\nTN (µg/L)', 'Depth (m)', 'Absolute latitude', 'Growing-season\nChla (µg/L)', ]

    g = sns.PairGrid(used_data_df[cols], corner=True, diag_sharey=False)
    g.map_lower(
        sns.scatterplot,
        s=10,
        alpha=0.4,
        color=scat_c,
        edgecolor=scat_c
    )
    g.map_lower(
        draw_lowess, color=line_c
    )
    g.map_diag(
        sns.kdeplot,
        fill=True,
        color=line_c,
        alpha=1,
        linewidth=1
    )
    max_vals = []
    for col in cols:
        from scipy.stats import gaussian_kde
        data = used_data_df[col].dropna().values
        kde = gaussian_kde(data)
        x_eval = np.linspace(data.min(), data.max(), 1000)
        y_eval = kde(x_eval)
        max_vals.append(y_eval.max())
    for ax, ymax in zip(g.diag_axes, max_vals):
        ax.set_ylim(0, ymax * 1.25)

    for ax in g.axes.flatten():
        if ax is not None:
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
            ax.tick_params(
                axis='both',
                which='major',
                width=.5,
                length=3,
                labelsize=8,
                direction='in'
            )
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)

    g.fig.subplots_adjust(wspace=0.1, hspace=0.1)
    g.fig.set_size_inches(8, 6)
    plt.tight_layout()
    plt.savefig('Figs/FS WQ pair plot.png', dpi=300)


def fs_lake_wq_n_years():
    used_data_df = filter_lakes(n_year=1, t_depth=100, if_mean=False, if_filter=False)
    counts = used_data_df['Unique Lake'].value_counts()

    fig = plt.figure(figsize=(5, 3))
    count_values = counts.value_counts().sort_index()
    bax = brokenaxes(ylims=((0, 400), (1100, 1200)), hspace=.1, fig=fig)
    bax.bar(count_values.index, count_values.values, color=line_c, edgecolor='k', linewidth=.5)
    bax.set_xlabel('Length of record (years)')
    bax.set_ylabel('Number of lakes')
    bax.set_xticks(range(1, 41))
    bax.set_xlim(0, 40)
    plt.savefig('Figs/FS WQ record years.png')


def fs_lake_wq_heatmap():
    used_data_df = filter_lakes(n_year=1, t_depth=100, if_mean=False, if_filter=False)
    print('filter n lake', len(used_data_df))
    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Depth (m)'] = used_data_df['depth']
    used_data_df['LN (µg P/L)'] = used_data_df['ln']
    used_data_df['Growing-season\nTP (µg/L)'] = used_data_df['TP (µg/L)']
    used_data_df['Growing-season\nTN (µg/L)'] = used_data_df['TN (µg/L)']
    used_data_df['Growing-season\nChla (µg/L)'] = used_data_df['Chla (µg/L)']

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 10))

    cs = ['LN (µg P/L)', 'Chla (µg/L)']
    for i_c, c in enumerate(cs):
        plt.sca(axes[0][i_c])
        x, y, z = used_data_df['Depth (m)'], used_data_df['Absolute latitude'], used_data_df[c]
        X = np.column_stack([x, y, x * y])
        model = LinearRegression().fit(X, z)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        ZZ = model.predict(np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() * yy.ravel()]))
        ZZ = ZZ.reshape(xx.shape)

        plt.contourf(xx, yy, ZZ, levels=30, cmap='viridis')
        sc = plt.scatter(x, y, c=z, cmap='viridis', s=30, linewidths=.5, edgecolor='k')
        cbar = plt.colorbar(sc)
        cbar.set_label(c, fontsize=9)
        cbar.ax.tick_params(labelsize=8, width=.5, length=2)
        cbar.outline.set_linewidth(.5)
        for spine in axes[0][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[0][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Depth (m)", fontsize=9)
        plt.ylabel("Absolute latitude", fontsize=9)

    cs = ['Depth (m)', 'Absolute latitude']
    for i_c, c in enumerate(cs):
        plt.sca(axes[1][i_c])
        x, y, z = used_data_df['LN (µg P/L)'], used_data_df[c], used_data_df['Chla (µg/L)']
        X = np.column_stack([x, y, x * y])
        model = LinearRegression().fit(X, z)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        ZZ = model.predict(np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() * yy.ravel()]))
        ZZ = ZZ.reshape(xx.shape)

        plt.contourf(xx, yy, ZZ, levels=30, cmap='viridis')
        sc = plt.scatter(x, y, c=z, cmap='viridis', s=30, linewidths=.5, edgecolor='k')
        cbar = plt.colorbar(sc)
        cbar.set_label('Chla (µg/L)', fontsize=9)
        cbar.ax.tick_params(labelsize=8, width=.5, length=2)
        cbar.outline.set_linewidth(.5)
        for spine in axes[1][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[1][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel('LN (µg P/L)', fontsize=9)
        plt.ylabel(c, fontsize=9)

    plt.sca(axes[2][0])
    x, y, z = used_data_df['Depth (m)'], used_data_df['Absolute latitude'], np.log10(used_data_df['Chla (µg/L)']) / np.log10(used_data_df['TP (µg/L)'])
    X = np.column_stack([x, y, x * y])
    model = LinearRegression().fit(X, z)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    ZZ = model.predict(np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() * yy.ravel()]))
    ZZ = ZZ.reshape(xx.shape)

    plt.contourf(xx, yy, ZZ, levels=30, cmap='viridis')
    sc = plt.scatter(x, y, c=z, cmap='viridis', s=30, linewidths=.5, edgecolor='k')
    cbar = plt.colorbar(sc)
    cbar.set_label('Chla / TP (log µg Chla / log µg P)', fontsize=8)
    cbar.ax.tick_params(labelsize=8, width=.5, length=2)
    cbar.outline.set_linewidth(.5)
    for spine in axes[2][0].spines.values():
        spine.set_linewidth(.5)
    axes[2][0].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
    plt.grid(lw=.5, alpha=0.4)
    plt.xlabel("Depth (m)", fontsize=9)
    plt.ylabel("Absolute latitude", fontsize=9)

    plt.delaxes(axes[2][1])
    plt.tight_layout()
    plt.savefig('Figs/FS N2C factors.png', dpi=300)


def fs_lake_wq_data_var():
    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=False, if_filter=False)
    print('filter n lake', len(used_data_df))
    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['depth (m)'] = used_data_df['depth']

    mean_data_df = used_data_df.groupby('Unique Lake').mean()
    mean_data_df = calc_ln(mean_data_df)
    print(len(used_data_df), len(mean_data_df))

    cs = ['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)']
    std_df = pd.DataFrame(index=mean_data_df.index, columns=cs, dtype=np.float64)
    for lake in mean_data_df.index:
        for v in cs:
            std_df.loc[lake, v] = np.std(used_data_df[used_data_df['Unique Lake'] == lake][v].values)

    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(3, 3, figure=fig, wspace=0.4, hspace=0.25, left=0.07, right=0.975, bottom=0.05, top=0.975)
    ax01, ax02, ax03 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax11, ax12, ax13 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    gs_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], wspace=0.25)
    ax20 = fig.add_subplot(gs_bottom[0, 0])
    ax21 = fig.add_subplot(gs_bottom[0, 1])
    axes = [[ax01, ax02, ax03], [ax11, ax12, ax13], [ax20, ax21]]

    for i_c, c in enumerate(cs):
        plt.sca(axes[0][i_c])
        plt.scatter(mean_data_df[c], std_df[c], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)

        lowess_result = sm.nonparametric.lowess(std_df[c], mean_data_df[c], frac=.7)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2)
        for spine in axes[0][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[0][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Mean growing-season {}".format(c), fontsize=9)
        plt.ylabel('SD of growing-season {}'.format(c), fontsize=9)

    cv_df = pd.read_csv('Results/cv.csv', header=0, index_col=0)
    for i_c, c in enumerate(cs):
        plt.sca(axes[1][i_c])
        plt.scatter(cv_df['n_year_Ch'], cv_df['rse_{}'.format(c[:2])] * 100, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)

        lowess_result = sm.nonparametric.lowess(cv_df['rse_{}'.format(c[:2])] * 100, cv_df['n_year_Ch'], frac=.5)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2)
        for spine in axes[1][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[1][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Number of years", fontsize=9)
        plt.ylabel('RSE of mean growing-season {}'.format(c.replace('µg/L', '%')), fontsize=9)

    for i_c, (c_x, c_y) in enumerate([['TP (µg/L)', 'Chla (µg/L)'], ['TN (µg/L)', 'Chla (µg/L)']]):
        plt.sca(axes[2][i_c])
        plt.scatter(cv_df['rse_{}'.format(c_x[:2])] * 100, cv_df['rse_{}'.format(c_y[:2])] * 100, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        plt.plot([0, 30], [50, 50], c='r', linewidth=2)
        plt.plot([30, 30], [0, 50], c='r', linewidth=2)
        for spine in axes[2][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[2][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.xlim(0, 60)
        plt.ylim(0, None)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel('RSE of mean growing-season {}'.format(c_x.replace('µg/L', '%')), fontsize=9)
        plt.ylabel('RSE of mean growing-season {}'.format(c_y.replace('µg/L', '%')), fontsize=9)

    plt.tight_layout()
    plt.savefig('Figs/FS WQ vars.png', dpi=300)


def fs_lake_wq_data_var_2():
    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=False, if_filter=False)
    print('filter n lake', len(used_data_df))
    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['depth (m)'] = used_data_df['depth']

    mean_data_df = used_data_df.groupby('Unique Lake').mean()
    mean_data_df = calc_ln(mean_data_df)
    print(len(used_data_df), len(mean_data_df))

    cs = ['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)']
    std_df = pd.DataFrame(index=mean_data_df.index, columns=cs, dtype=np.float64)
    cv_df = pd.DataFrame(index=mean_data_df.index, columns=cs, dtype=np.float64)
    for lake in mean_data_df.index:
        for v in cs:
            std_df.loc[lake, v] = np.std(used_data_df[used_data_df['Unique Lake'] == lake][v].values)
            cv_df.loc[lake, v] = std_df.loc[lake, v] / mean_data_df.loc[lake, v]

    fig = plt.figure(figsize=(9, 15))
    gs = GridSpec(5, 3, figure=fig, wspace=0.4, hspace=0.25, left=0.07, right=0.975, bottom=0.05, top=0.975)
    ax01, ax02, ax03 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax11, ax12, ax13 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    gs_bottom = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2:, :], wspace=0.25)
    ax21, ax31, ax41 = fig.add_subplot(gs_bottom[0]), fig.add_subplot(gs_bottom[1]), fig.add_subplot(gs_bottom[2])
    axes = [[ax01, ax02, ax03], [ax11, ax12, ax13], [ax21, ax31, ax41]]

    for i_c, c in enumerate(cs):
        plt.sca(axes[0][i_c])
        plt.scatter(mean_data_df[c], std_df[c], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(std_df[c], mean_data_df[c], frac=.7)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2)
        for spine in axes[0][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[0][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Mean growing-season {}".format(c), fontsize=9)
        plt.ylabel('SD of\nmean growing-season {}'.format(c), fontsize=9)

    for i_c, c in enumerate(cs):
        plt.sca(axes[1][i_c])
        plt.scatter(mean_data_df[c], cv_df[c], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(cv_df[c], mean_data_df[c], frac=.8)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2)
        for spine in axes[1][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[1][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Mean growing-season {}".format(c), fontsize=9)
        plt.ylabel('CV of\nmean growing-season {}'.format(c), fontsize=9)

    cv_df = pd.read_csv('Results/cv.csv', header=0, index_col=0)
    for i_c, c in enumerate(cs):
        plt.sca(axes[2][i_c])
        plt.scatter(cv_df['n_year_Ch'], cv_df['rse_{}'.format(c[:2])] * 100, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)

        t_rse = 50
        plt.plot([0, 40], [t_rse, t_rse], ls='-', c='r', linewidth=2)
        for spine in axes[2][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[2][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.xlim([4.5, 40])
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Length of record (years)", fontsize=9)
        plt.ylabel('RSE of\nmean growing-season {}'.format(c.replace('µg/L', '%')), fontsize=9)

    text_dic = {ax01: 'a', ax02: 'b', ax03: 'c', ax11: 'd', ax12: 'e', ax13: 'f', ax21: 'g', ax31: 'h', ax41: 'j', }
    for ax, text in text_dic.items():
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig('Figs/FS WQ var2.png', dpi=300)


def fs_lake_n2c_reg():
    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=True, if_filter=False)
    used_data_df = filter_lakes_with_rse(used_data_df, t_chl=0.5)
    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['depth (m)'] = used_data_df['depth']

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    cs = ['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)']
    for i_c, c in enumerate(cs[1:]):
        plt.sca(axes[0][i_c])
        plt.scatter(used_data_df[c], used_data_df['Chla (µg/L)'], alpha=.5, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        slope, intercept, r_value, p_value, std_err = linregress(used_data_df[c], used_data_df['Chla (µg/L)'])
        plt.plot(used_data_df[c], used_data_df[c] * slope + intercept, line_c, linewidth=2, label='OLS regression (R$^2$={:.2f}, n={})'.format(r_value**2, len(used_data_df)))
        for spine in axes[0][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[0][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Long-term growing-season {}".format(c), fontsize=9)
        plt.ylabel("Long-term growing-season {}".format('Chla (µg/L)'), fontsize=9)
        plt.legend(fontsize=9, loc='upper left')

    plt.sca(axes[1][0])
    p_df = used_data_df[used_data_df['ln_t'] == 0]
    plt.scatter(p_df['TP (µg/L)'], p_df['TN:TP'], alpha=.5, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5, label='P limiting (n={})'.format(len(p_df)))
    tp = np.sort(used_data_df['TP (µg/L)'].values)
    plt.plot(tp, calc_std_tn(tp) / tp, c='r', lw=2, label='Critical TN:TP')
    n_df = used_data_df[used_data_df['ln_t'] == 1]
    plt.scatter(n_df['TP (µg/L)'], n_df['TN:TP'], alpha=.5, c='#723f30', edgecolors='#723f30', s=30, linewidths=.5, label='N limiting (n={})'.format(len(n_df)))
    for spine in axes[1][0].spines.values():
        spine.set_linewidth(.5)
    axes[1][0].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
    plt.ylim(None, 100)
    plt.grid(lw=.5, alpha=0.4)
    plt.xlabel("Long-term growing-season {}".format('TP (µg/L)'), fontsize=9)
    plt.ylabel("Long-term growing-season {}".format('TN:TP'), fontsize=9)
    plt.legend(fontsize=9, loc='upper left')

    plt.sca(axes[1][1])
    c = 'ln'
    plt.scatter(used_data_df[c], used_data_df['Chla (µg/L)'], alpha=.5, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
    slope, intercept, r_value, p_value, std_err = linregress(used_data_df[c], used_data_df['Chla (µg/L)'])
    plt.plot(used_data_df[c], used_data_df[c] * slope + intercept, line_c, linewidth=2, label='OLS regression (R$^2$={:.2f}, n={})'.format(r_value**2, len(used_data_df)))

    for spine in axes[1][1].spines.values():
        spine.set_linewidth(.5)
    axes[1][1].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
    plt.grid(lw=.5, alpha=0.4)
    plt.xlabel("Long-term growing-season {}".format('LN (µg P/L)'), fontsize=9)
    plt.ylabel("Long-term growing-season {}".format('Chla (µg/L)'), fontsize=9)
    plt.legend(fontsize=9, loc='upper left')

    text_dic = {axes[0][0]: 'a', axes[0][1]: 'b', axes[1][0]: 'a', axes[1][1]: 'b'}
    for ax, text in text_dic.items():
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig('Figs/FS N2C.png', dpi=300)


def fs_lake_n2c_reg_factors():
    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=True, if_filter=False)
    used_data_df = filter_lakes_with_rse(used_data_df, t_chl=0.5)

    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Depth (m)'] = used_data_df['depth']
    used_data_df['LN (µg P/L)'] = used_data_df['ln']
    used_data_df['Long-term growing-season Chla (µg/L)'] = used_data_df['Chla (µg/L)']
    used_data_df['Long-term growing-season LN (µg P/L)'] = used_data_df['LN (µg P/L)']

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 10))
    cs = ['Long-term growing-season LN (µg P/L)', 'Long-term growing-season Chla (µg/L)']
    for i_c, c in enumerate(cs):
        plt.sca(axes[0][i_c])
        x, y, z = used_data_df['Depth (m)'], used_data_df['Absolute latitude'], used_data_df[c]
        X = np.column_stack([x, y, x * y])
        model = LinearRegression().fit(X, z)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        ZZ = model.predict(np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() * yy.ravel()]))
        ZZ = ZZ.reshape(xx.shape)

        plt.contourf(xx, yy, ZZ, levels=30, cmap='viridis')
        sc = plt.scatter(x, y, c=z, cmap='viridis', s=30, linewidths=.5, edgecolor='k')
        cbar = plt.colorbar(sc)
        cbar.set_label(c, fontsize=9)
        cbar.ax.tick_params(labelsize=8, width=.5, length=2)
        cbar.outline.set_linewidth(.5)
        for spine in axes[0][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[0][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Depth (m)", fontsize=9)
        plt.ylabel("Absolute latitude", fontsize=9)

    cs = ['Depth (m)', 'Absolute latitude']
    for i_c, c in enumerate(cs):
        plt.sca(axes[1][i_c])
        x, y, z = used_data_df['Long-term growing-season LN (µg P/L)'], used_data_df[c], used_data_df['Long-term growing-season Chla (µg/L)']
        X = np.column_stack([x, y, x * y])
        model = LinearRegression().fit(X, z)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        ZZ = model.predict(np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() * yy.ravel()]))
        ZZ = ZZ.reshape(xx.shape)

        plt.contourf(xx, yy, ZZ, levels=30, cmap='viridis')
        sc = plt.scatter(x, y, c=z, cmap='viridis', s=30, linewidths=.5, edgecolor='k')
        cbar = plt.colorbar(sc)
        cbar.set_label('Long-term growing-season Chla (µg/L)', fontsize=9)
        cbar.ax.tick_params(labelsize=8, width=.5, length=2)
        cbar.outline.set_linewidth(.5)
        for spine in axes[1][i_c].spines.values():
            spine.set_linewidth(.5)
        axes[1][i_c].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel('Long-term growing-season LN (µg P/L)', fontsize=9)
        plt.ylabel(c, fontsize=9)

    plt.sca(axes[2][0])
    x, y, z = used_data_df['Depth (m)'], used_data_df['Absolute latitude'], np.log10(used_data_df['Long-term growing-season Chla (µg/L)']) / np.log10(used_data_df['Long-term growing-season LN (µg P/L)'])
    X = np.column_stack([x, y, x * y])
    model = LinearRegression().fit(X, z)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    ZZ = model.predict(np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() * yy.ravel()]))
    ZZ = ZZ.reshape(xx.shape)

    plt.contourf(xx, yy, ZZ, levels=30, cmap='viridis')
    sc = plt.scatter(x, y, c=z, cmap='viridis', s=30, linewidths=.5, edgecolor='k')
    cbar = plt.colorbar(sc)
    cbar.set_label('Long-term growing-season Chla / LN \n(log µg Chla / log µg P)', fontsize=8)
    cbar.ax.tick_params(labelsize=8, width=.5, length=2)
    cbar.outline.set_linewidth(.5)
    for spine in axes[2][0].spines.values():
        spine.set_linewidth(.5)
    axes[2][0].tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
    plt.grid(lw=.5, alpha=0.4)
    plt.xlabel("Depth (m)", fontsize=9)
    plt.ylabel("Absolute latitude", fontsize=9)

    text_dic = {axes[0][0]: 'a', axes[0][1]: 'b', axes[1][0]: 'c', axes[1][1]: 'd', axes[2][0]: 'e'}
    for ax, text in text_dic.items():
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.delaxes(axes[2][1])
    plt.tight_layout()
    plt.savefig('Figs/FS N2C factors.png', dpi=300)
    plt.show()


def fs_response_samples():
    N = 250
    mean_data_df = filter_lakes(n_year=5, t_depth=25, if_mean=True, if_filter=False)

    lat_mini, length, step, n, n_sam = -4, 5, .25, 80, N
    lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
    xs, ys, rs, fs, ics, gis, gls, gips, ns, ats, wts = [], [], [], [], [], [], [], [], [], [], []
    for lat_min, lat_max in lat_windows:
        for i in range(n_sam):
            samples = sample_lakes_in_depth_band(mean_data_df, lat_min, lat_max, n=30, ln_bins=10, random_state=911 + i)
            if len(samples) >= 10:
                x, y = samples['ln'], samples['Chla (µg/L)']
                x_with_const = sm.add_constant(x)
                glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                y_pred = glm_model.predict(x_with_const)
                slope, p_value = glm_model.params.loc['ln'], glm_model.pvalues.loc['ln']
                if p_value < 0.05:
                    fs.append(r2_score(y, y_pred))
                    ys.append(slope)
                    xs.append(samples['depth'].mean())

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 10))
    plt.sca(ax1)
    xs = np.array(xs)
    ys = np.array(ys)
    plt.scatter(xs[xs < 6], ys[xs < 6], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
    plt.scatter(xs[xs >= 6], ys[xs >= 6], alpha=.25, c=depth_color, edgecolors=depth_color, s=30, linewidths=.5)
    lowess_result = sm.nonparametric.lowess(ys, xs, frac=.5)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)

    plt.xlim([0, 20])
    plt.xticks([0, 5, 10, 15, 20])
    plt.ylim([.2, .8])
    plt.yticks([.2, .4, .6, .8])
    plt.xlabel('Lake average depth (m)', fontsize=9)
    plt.ylabel('Nutrient-Chl$a$ slope (µg Chl$a$ / µg P)', fontsize=9)
    ax1.text(0.95, 0.95, "a", transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.sca(ax2)
    plt.scatter(xs[xs < 6], fs[xs < 6], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
    plt.scatter(xs[xs >= 6], fs[xs >= 6], alpha=.25, c=depth_color, edgecolors=depth_color, s=30, linewidths=.5)
    lowess_result = sm.nonparametric.lowess(fs, xs, frac=.4)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=101)

    plt.ylim([0, 1])
    plt.yticks([0, .25, .5, 0.75, 1])
    plt.xlim([0, 20])
    plt.xticks([0, 5, 10, 15, 20])
    plt.xlabel('Lake average depth (m)', fontsize=9)
    plt.ylabel('R$^2$', fontsize=9)
    ax2.text(0.95, 0.95, "b", transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    colors = [scat_c, depth_color]
    line_colors = ['#1e4350', '#3b3b3b']
    for i_t, t_depth in enumerate([6, 30]):
        used_data_df = filter_lakes(n_year=5, t_depth=t_depth)
        used_data_df['Unique Lake'] = used_data_df.index
        valid_lakes = used_data_df.index
        lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
        lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
        cli_lake_ids = [valid_lake for valid_lake in valid_lakes if valid_lake in lake_climate_df.index]
        used_lake_cli_df = lake_climate_df.loc[cli_lake_ids]

        lat_mini, length, step, n, n_sam = 15, 15, .5, 70, N
        lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
        xs, ys, fs, ics, ats, wts, dps = [], [], [], [], [], [], []
        for lat_min, lat_max in lat_windows:
            for i in range(n_sam):
                samples = sample_lakes_in_lat_band(used_data_df, lat_min, lat_max, n=30, tp_bins=10, random_state=911 + i + int(lat_min))
                if len(samples) >= 10:
                    x, y = samples['ln'], samples['Chla (µg/L)']
                    x_with_const = sm.add_constant(x)
                    glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                    y_pred = glm_model.predict(x_with_const)
                    slope, intercept, p_value = glm_model.params.loc['ln'], glm_model.params.loc['const'], \
                        glm_model.pvalues.loc['ln']
                    if p_value < 0.05:
                        xs.append(samples['Latitude'].mean())
                        ys.append(slope)
                        fs.append(r2_score(y, y_pred))
                        ics.append(intercept)
                        ats.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'temperature_2m']))
                        wts.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'lake_mix_layer_temperature']))
                        dps.append(samples['depth'].mean())

        plt.sca(ax3)
        plt.scatter(xs, ys, alpha=.25, c=colors[i_t], edgecolors=colors[i_t], s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(ys, xs, frac=.125)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_colors[i_t], linewidth=2, zorder=100)
        plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
        plt.yticks([0, .5, 1, 1.5])
        plt.xlabel('Absolute latitude', fontsize=9)
        plt.ylabel('Nutrient-Chl$a$ slope (µg Chl$a$ / µg P)', fontsize=9)
        ax3.text(0.95, 0.95, "c", transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

        plt.sca(ax4)
        xs = np.array(xs)
        fs = np.array(fs)
        plt.scatter(xs, fs, alpha=.25, c=colors[i_t], edgecolors=colors[i_t], s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(fs, xs, frac=.4)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_colors[i_t], linewidth=2, zorder=100)
        plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
        plt.ylim([0, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.xlabel('Absolute latitude', fontsize=9)
        plt.ylabel('R$^2$', fontsize=9)
        ax4.text(0.95, 0.95, "d", transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

        plt.sca(ax5)
        plt.scatter(xs, wts, alpha=.25, c=colors[i_t], edgecolors=colors[i_t], s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(wts, xs, frac=.4)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_colors[i_t], linewidth=2, zorder=100)
        plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
        plt.xlabel('Absolute latitude', fontsize=9)
        plt.ylabel('Growing-season temperature (°C)', fontsize=9)
        ax5.text(0.95, 0.95, "e", transform=ax5.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

        plt.sca(ax6)
        plt.scatter(xs, dps, alpha=.25, c=colors[i_t], edgecolors=colors[i_t], s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(dps, xs, frac=.4)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_colors[i_t], linewidth=2, zorder=100)
        plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
        plt.xlabel('Absolute latitude', fontsize=9)
        plt.ylabel('Lake average depth (m)', fontsize=9)
        ax6.text(0.95, 0.95, "f", transform=ax6.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig('Figs/FS Samples along depth and latitude.png')


def fs_bayesian_model_selection_fitness():
    waic_path = 'Results/waic.csv'
    if os.path.exists(waic_path):
        waic_df = pd.read_csv(waic_path, index_col=0, header=0)
    else:
        used_data_df = filter_lakes()

        used_data_df['Unique Lake'] = used_data_df.index
        lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
        lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
        used_lake_ids = used_data_df.index.intersection(lake_climate_df.index)
        used_lake_cli_df = lake_climate_df.loc[used_lake_ids]
        merge_df = pd.concat([used_data_df, used_lake_cli_df], axis=1)
        print(merge_df.columns)

        y = merge_df['Chla (µg/L)'].values
        all_combinations = []
        all_combinations.extend([['Latitude'], ['surface_solar_radiation_downwards_sum'], ['Latitude', 'surface_solar_radiation_downwards_sum', 'temperature_2m'], ['temperature_2m']])
        all_combinations.extend([['TN (µg/L)'], ['TP (µg/L)'], ['TN (µg/L)', 'TP (µg/L)', 'ln'], ['ln']])
        all_combinations.extend([['mix_ln_at'], ['temperature_2m', 'ln'], ['ln', 'temperature_2m', 'mix_ln_at']])

        # Scale
        merge_df['Latitude'] /= 60
        merge_df['ln'] /= 100
        merge_df['TN (µg/L)'] /= 3000
        merge_df['TP (µg/L)'] /= 100
        merge_df['temperature_2m'] /= 40
        merge_df['surface_solar_radiation_downwards_sum'] /= 800
        merge_df['mix_ln_at'] = merge_df['ln'] * merge_df['temperature_2m']

        xtick_dic = {
            'surface_solar_radiation_downwards_sum': 'Rad',
            'Latitude': 'Lat',
            'temperature_2m': 'Temp',
            'TN (µg/L)': 'TN',
            'TP (µg/L)': 'TP',
            'ln': 'LN',
            'mix_ln_at': 'Temp×LN',
        }

        waic_results = []
        for combo in all_combinations:
            X = merge_df[list(combo)].values
            n_features = X.shape[1]

            with pm.Model() as model:
                beta0 = pm.Normal("beta0", mu=1, sigma=10)
                betas = pm.Normal('betas', mu=1, sigma=10, shape=n_features)
                mu = beta0 + pm.math.dot(X, betas)
                sigma = pm.Exponential("sigma", 10)
                y_obs = pm.LogNormal("y_obs", mu=pm.math.log(mu), sigma=sigma, observed=y)
                trace = pm.sample(2000, tune=2000, return_inferencedata=True, target_accept=0.9, random_seed=119, chains=4)
                pm.compute_log_likelihood(trace)
                ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
                y_pred_samples = ppc.posterior_predictive["y_obs"].values

            if combo in [['ln', 'temperature_2m'], ['ln', 'temperature_2m', 'mix_ln_at'], ['temperature_2m', 'mix_ln_at'], ['ln', 'mix_ln_at'], ['mix_ln_at']]:
                betas_samples = trace.posterior["betas"].values.flatten().reshape(n_features, -1)
                idx_mix = combo.index('mix_ln_at') if 'mix_ln_at' in combo else -1
                beta_mix_samples = 0 if idx_mix < 0 else betas_samples[idx_mix]
                idx_ln = combo.index('ln') if 'ln' in combo else -1
                beta_ln_samples = 0 if idx_ln < 0 else betas_samples[idx_ln]
                temps = np.linspace(0, 40, 200) / 40
                beta_effects = np.array(
                    [beta_ln_samples + beta_mix_samples * T for T in temps])  # shape = [n_temp, n_samples]
                low_eff, high_eff = np.percentile(beta_effects, [5, 97.5], axis=1)
                sig_mask = (low_eff > 0)
                min_t = temps[sig_mask]
                if len(min_t) > 0:
                    min_t = min_t[0] * 40
                else:
                    min_t = -1
            else:
                min_t = -1
            waic = az.waic(trace)
            form = 'Chla ~ {}'.format(xtick_dic[combo[0]])
            if len(combo) > 1:
                for c in combo[1:]:
                    form += ' + {}'.format(xtick_dic[c])
            waic_results.append({'combo': combo, 'waic': waic.elpd_waic, 'se': waic.se, 'p': waic.p_waic, 'r2': max(r2_score(y, np.mean(y_pred_samples, axis=(0, 1))), 0), 'min_t': min_t, 'form': form})
        waic_df = pd.DataFrame(waic_results)
        print(waic_df)
        waic_df.to_csv('Results/waic.csv', index=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    plt.sca(ax1)
    break_idx_ls = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]

    for i, idx in enumerate(break_idx_ls):
        sub_df = waic_df.loc[idx]
        values = sub_df['r2'].values
        plt.plot(sub_df.index, values, c=scat_c, lw=1, alpha=.5, ls='--')
        sc = plt.scatter(sub_df.index, values, alpha=.8, c=scat_c, edgecolors='k', s=30, linewidths=.5, label='R$^2$')
    plt.ylabel('R$^2$')
    plt.xticks(range(len(waic_df)), waic_df['form'].values, rotation=90)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylim(0, 1)
    legend1 = ax1.legend(handles=[sc], loc='upper left')
    ax1.add_artist(legend1)
    ax_ty = plt.gca().twinx()
    plt.sca(ax_ty)
    for i, idx in enumerate(break_idx_ls):
        sub_df = waic_df.loc[idx]
        values = sub_df['waic'].values
        plt.plot(sub_df.index, values, c=line_c, lw=1, alpha=.5, ls='--')
        sc = plt.scatter(sub_df.index, values, alpha=.8, c=line_c, edgecolors='k', s=30, linewidths=.5, label='WAIC')
    eb = plt.errorbar(waic_df.index, waic_df['waic'].values, yerr=waic_df['se'].values, fmt='', linestyle='', elinewidth=1, capsize=3, alpha=0.6, ecolor=line_c, label="SE of WAIC")
    plt.ylabel('WAIC')
    plt.yticks([-1600, -1500, -1400, -1300, -1200, -1100])
    plt.ylim(-1600, -1200)
    legend2 = ax_ty.legend(handles=[sc, eb], loc='lower right')
    ax_ty.add_artist(legend2)
    ax1.text(0.18, 0.2, 'Meteorological models', fontsize=9, ha='center', va='center', color='k', transform=ax1.transAxes)
    ax1.text(0.55, 0.45, 'Nutrient models', fontsize=9, ha='center', va='center', color='k', transform=ax1.transAxes)
    ax1.text(0.87, 0.7, 'Combined models', fontsize=9, ha='center', va='center', color='k', transform=ax1.transAxes)

    used_data_df = filter_lakes()
    used_data_df['Unique Lake'] = used_data_df.index
    lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
    lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
    used_lake_ids = used_data_df.index.intersection(lake_climate_df.index)
    used_lake_cli_df = lake_climate_df.loc[used_lake_ids]
    merge_df = pd.concat([used_data_df, used_lake_cli_df], axis=1)

    y = merge_df['Chla (µg/L)'].values
    combo = ['ln', 'temperature_2m', 'mix_ln_at']
    merge_df['Latitude'] /= 60
    merge_df['ln'] /= 100
    merge_df['TN (µg/L)'] /= 3000
    merge_df['TP (µg/L)'] /= 100
    merge_df['temperature_2m'] /= 40
    merge_df['surface_solar_radiation_downwards_sum'] /= 800
    merge_df['mix_ln_at'] = merge_df['ln'] * merge_df['temperature_2m']

    xtick_dic = {
        'surface_solar_radiation_downwards_sum': 'Rad',
        'Latitude': 'Lat',
        'temperature_2m': 'Temp',
        'TN (µg/L)': 'TN',
        'TP (µg/L)': 'TP',
        'ln': 'LN',
        'mix_ln_at': 'Temp×LN',
    }
    X = merge_df[list(combo)].values
    n_features = X.shape[1]

    with pm.Model() as model:
        # 先验
        beta0 = pm.Normal("beta0", mu=0, sigma=10)
        betas = pm.Normal('betas', mu=0, sigma=10, shape=n_features)
        mu = beta0 + pm.math.dot(X, betas)
        sigma = pm.Exponential("sigma", 10)
        y_obs = pm.LogNormal("y_obs", mu=pm.math.log(mu), sigma=sigma, observed=y)
        trace = pm.sample(2000, tune=2000, return_inferencedata=True, target_accept=0.9, random_seed=119, chains=4)
        pm.compute_log_likelihood(trace)
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
        y_pred_samples = ppc.posterior_predictive["y_obs"].values

    y_pred = np.mean(y_pred_samples, axis=(0, 1))
    y_pred_lower = np.percentile(y_pred_samples, 5, axis=(0, 1))
    y_pred_upper = np.percentile(y_pred_samples, 95, axis=(0, 1))
    r2 = r2_score(y, y_pred)
    form = 'Chla ~ {}'.format(xtick_dic[combo[0]])
    if len(combo) > 1:
        for c in combo[1:]:
            form += ' + {}'.format(xtick_dic[c])

    plt.sca(ax2)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '-', lw=0.5, alpha=1, color='k', label="1:1 line")
    plt.scatter(y, y_pred, c=scat_c, edgecolors='k', s=30, linewidths=.5, alpha=.6, label='Posterior mean')
    plt.errorbar(y, y_pred, yerr=[y_pred - y_pred_lower, y_pred_upper - y_pred], fmt='', linestyle='', elinewidth=1, capsize=0, alpha=0.3, ecolor=scat_c, label="95% predictive interval")
    plt.xlabel("Measured long-term growing-season Chla (µg/L)")
    plt.ylabel("Predicted long-term growing-season Chla (µg/L)")
    plt.xlim(0, 80)
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    ax2.text(0.05, 0.95, 'Combined model: {}\nR$^2$: {:.2f} (n={})'.format(form, r2, len(y)), fontsize=9, ha='left', va='top', color='k', transform=ax2.transAxes)

    text_dic = {ax1: 'a', ax2: 'b'}
    for ax, text in text_dic.items():
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig('Figs/FS Model selection and fitness.png')


def fs_bayesian_model_jn_analysis(if_plot=True):
    used_data_df = filter_lakes()
    used_data_df = calc_ln(used_data_df)
    used_data_df['Unique Lake'] = used_data_df.index
    lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
    lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
    used_lake_ids = used_data_df.index.intersection(lake_climate_df.index)
    used_lake_cli_df = lake_climate_df.loc[used_lake_ids]
    merge_df = pd.concat([used_data_df, used_lake_cli_df], axis=1)

    X_names = ['ln', 'temperature_2m', 'mix_ln_at']
    merge_df['ln'] = merge_df['ln']
    merge_df['temperature_2m'] = merge_df['temperature_2m']
    merge_df['mix_ln_at'] = merge_df['ln'] * merge_df['temperature_2m']
    X = merge_df[X_names].values
    y = merge_df['Chla (µg/L)'].values

    with pm.Model() as model_lognormal:
        beta0 = pm.Normal("beta0", mu=0, sigma=10)
        betas = pm.Normal('betas', mu=0, sigma=10, shape=len(X_names))
        mu = beta0 + pm.math.dot(X, betas)
        sigma = pm.Exponential("sigma", 10)
        y_obs = pm.LogNormal("y_obs", mu=pm.math.log(mu), sigma=sigma, observed=y)
        trace = pm.sample(2000, tune=2000, return_inferencedata=True, target_accept=0.9, random_seed=119, chains=4)
        pm.compute_log_likelihood(trace)
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
        y_pred_samples = ppc.posterior_predictive["y_obs"].values

    y_pred_mean = np.mean(y_pred_samples, axis=(0, 1))
    y_pred_lower = np.percentile(y_pred_samples, 10, axis=(0, 1))  # 2.5% quantile
    y_pred_upper = np.percentile(y_pred_samples, 90, axis=(0, 1))  # 97.5% quantile

    beta0_samples = trace.posterior["beta0"].values.flatten()  # shape: (n_samples,)
    betas_samples = trace.posterior["betas"].values.flatten().reshape(len(X_names), -1)
    betas_std = np.std(betas_samples, axis=1)

    temps = np.linspace(0, 40, 200)
    idx_mix = X_names.index('mix_ln_at') if 'mix_ln_at' in X_names else -1
    beta_mix_samples = 0 if idx_mix < 0 else betas_samples[idx_mix]
    idx_ln = X_names.index('ln') if 'ln' in X_names else -1
    beta_ln_samples = 0 if idx_ln < 0 else betas_samples[idx_ln]
    beta_effects = np.array([beta_ln_samples + beta_mix_samples * T for T in temps])  # shape = [n_temp, n_samples]
    low_eff, med_eff, high_eff = np.percentile(beta_effects, [2.5, 50, 97.5], axis=1)
    mean_eff = np.mean(beta_effects, axis=1)
    sig_mask = (low_eff > 0)
    min_t = temps[sig_mask]
    min_t = min_t[0] if len(min_t) > 0 else -1
    slope_mean, intercept_mean, r_value, p_value, std_err = linregress(temps, mean_eff)

    if if_plot:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        plt.plot([-2, 42], [0, 0], ls='-', c='k', lw=.5)
        plt.plot(temps, mean_eff, lw=2, c=line_c, alpha=1, label='Posterior mean')
        plt.fill_between(temps, low_eff, high_eff, color=line_c, alpha=.25, label='95% posterior interval')
        for spine in ax.spines.values():
            spine.set_linewidth(.5)
        ax.tick_params(axis='both', which='major', width=.5, length=2, labelsize=8)
        plt.grid(lw=.5, alpha=0.4)
        plt.xlabel("Growing-season temperature (°C)", fontsize=9)
        plt.ylabel("Nutrient-Chla slope (µg Chla / µg P)", fontsize=9)
        plt.xlim(-2, 42)
        plt.ylim(-.5, 2)
        plt.plot([min_t, min_t], [0, -.5], ls='-', c='r', lw=1)
        plt.scatter(min_t, 0, s=30, edgecolors='r', c='r', label='Temperature threshold of significance', zorder=100)
        form = 'Chla = $β_0$ + $β_1$LN + $β_2$Temp + $β_3$Temp×LN'
        slope = '$∂$Chla/$∂$LN = $β_1$ + $β_3$Temp'
        ax.text(0.02, 0.85, 'Combined model: {}\nNutrient-Chla slope: {}'.format(form, slope), fontsize=9, ha='left', va='top', color='k', transform=ax.transAxes)
        plt.legend(fontsize=9, loc='upper left')
        plt.tight_layout()
        plt.savefig('Figs/FS J-N.png', dpi=300)
        plt.show()
    else:
        pass
    return temps, mean_eff, low_eff, high_eff, min_t


def fs_global_lake_map():
    data_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col='Hylak_id')
    chla_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_mean_bayes.csv', index_col='Hylak_id')
    data_df = pd.concat([data_df, chla_df], axis=1)
    shallow_data_df = data_df[data_df['depth'] < 6]
    filter_data_df = data_df.dropna(subset=['ln', '2020'])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [4, 3]})
    ax1, ax2 = axes

    m = Basemap(projection='cyl', resolution='i', ax=ax1)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='lightgray', lake_color='lightgray', alpha=.5)
    m.drawmapboundary(fill_color='none', linewidth=0.5)
    parallels = np.arange(-80, 81, 20)
    meridians = np.arange(-180, 181, 60)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')

    values = data_df['depth']
    norm = mpl.colors.Normalize(vmin=0, vmax=20, clip=True)
    x, y = m(data_df['Pour_long'].values, data_df['Pour_lat'].values)
    scatter = m.scatter(
        x, y,
        c=values,
        cmap='viridis',
        norm=norm,
        marker='s',
        s=.1,
        edgecolor='none',
        alpha=1,
    )
    ax1.set_aspect('auto')
    cb = m.colorbar(scatter, location='right', size="2%",pad='2%')
    cb.set_label("Lake average depth (m)")
    cb.set_ticks([0, 5, 10, 15, 20])
    cb.set_ticklabels(['0', '5', '10', '15', '>20'], fontsize=8)

    ax_inner = ax1.inset_axes([0.08, 0.12, 0.15, 0.4], zorder=100)
    shape, loc, scale = stats.lognorm.fit(data_df['depth'], floc=0)
    x = np.linspace(data_df['depth'].min(), data_df['depth'].max(), 2000)
    cdf = stats.lognorm.cdf(x, shape, loc=loc, scale=scale)
    ax_inner.plot(x, cdf, '#bb5090', linewidth=2)
    ax_inner.set_xlim(0, 20)
    ax_inner.set_ylim(0, 1.1)
    ax_inner.set_xticks([0, 5, 10, 15, 20])
    ax_inner.set_yticks([0., 0.25, .5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax_inner.set_ylabel('Cumulative proportion of lakes (%)', fontsize=7)
    ax_inner.set_xlabel('Average depth (m)', fontsize=7)
    ax_inner.tick_params(axis='both', which='major', width=.5, length=2, labelsize=6)

    plt.sca(ax2)
    width = 2
    lat_bins = np.arange(-90, 91, width)
    lat_labels = lat_bins[:-1]
    all_counts, _ = np.histogram(data_df['Pour_lat'], bins=lat_bins)
    shallow_counts, _ = np.histogram(shallow_data_df['Pour_lat'], bins=lat_bins)
    ax2.bar(lat_labels, all_counts, color='#B0C4DE', edgecolor='none', alpha=0.8, width=width * .75, label='Deeper lakes (average depth ≥ 6m)')
    ax2.bar(lat_labels, shallow_counts, color=scat_c, edgecolor='none', alpha=1, width=width * .75, label='Shallow lakes (average depth < 6m)')
    plt.xticks(range(-80, 81, 20), ['80°S', '60°S', '40°S', '20°S', '0°', '20°N', '40°N', '60°N', '80°N'])
    plt.xlim(-80, 80)
    ax2.set_yticks([0, 3e4, 6e4, 9e4, 12e4, 15e4], [0, 3, 6, 9, 12, 15])
    ax2.set_ylim([0, 15e4])
    ax2.set_ylabel('Number of lakes (×10$^4$)')
    ax2.set_xlabel('Latitude')
    plt.legend(loc='upper left')

    ax2_tx = ax2.twinx()
    ax2_tx.plot(lat_labels, shallow_counts / all_counts, lw=2, c=line_c, alpha=1, label='Proportion of shallow lakes')
    ax2_tx.set_ylim([0, 1])
    ax2_tx.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25], [0, 25, 50, 75, 100, 125])
    ax2_tx.set_ylabel('Proportion of shallow lakes (%)')
    plt.legend(loc='upper right')

    text_dic = {ax1: 'a', ax2: 'b'}
    for ax, text in text_dic.items():
        ax.text(0.05, 0.95 if text == 'a' else 0.80, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('Figs/FS Global lake map.png', dpi=300)


def fs_global_lake_future_temp_chla_map():
    data_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col='Hylak_id')
    chla_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_mean_bayes.csv', index_col='Hylak_id')
    chla_df.columns = [str(col) + '_chla' for col in chla_df.columns]
    temp_df = load_hydrolakes_cmip6_temp()
    temp_df.columns = [str(col) + '_temp' for col in temp_df.columns]
    data_df = pd.concat([data_df, chla_df, temp_df], axis=1)
    print('All lake num', len(data_df))

    cols = ['temp', 'chla']
    years = [2020, 2100]
    for c in cols:
        data_df['delta_{}'.format(c)] = data_df['{}_{}'.format(years[1], c)] - data_df['{}_{}'.format(years[0], c)]

    filter_data_df = data_df[data_df['depth'] < 6]
    filter_data_df = filter_data_df.dropna(subset=['ln', '2020_chla'])

    fig, axes = plt.subplots(4, 2, figsize=(8, 8))

    car_ticks_dic = {
        'year_temp': [0, 10, 20, 30, 40],
        'delta_temp': [2, 4, 6, 8],
        'year_chla': [0, 10, 20, 30, 40],
        'delta_chla': [0, 5, 10, 15, 20],
    }
    title_dic = {
        '2020_temp': 'Growing-season temperature (2020)',
        '2100_temp': 'Growing-season temperature (2100, SSP5-8.5)',
        'delta_temp': 'Change in temperature (2100 - 2020)',
        '2020_chla': 'Growing-season Chla (2020)',
        '2100_chla': 'Growing-season Chla (2100, SSP5-8.5)',
        'delta_chla': 'Change in Chla (2100 - 2020)',
    }
    cb_dic = {
        '2020_temp': 'Temperature (°C)',
        '2100_temp': 'Temperature (°C)',
        'delta_temp': 'ΔTemperature (°C)',
        '2020_chla': 'Chla (µg/L)',
        '2100_chla': 'Chla (µg/L)',
        'delta_chla': 'ΔChla (µg/L)',
    }
    kde_xticks_dic = {
        'year_temp': [0, 20, 40],
        'delta_temp': [2, 5, 8],
        'year_chla': [0, 50, 100],
        'delta_chla': [0, 20, 40],
    }
    color_dic = {
        '2020_temp': '#FED9A6',
        '2100_temp': 'r',
        'delta_temp': '#287D8E',
        '2020_chla': '#FED9A6',
        '2100_chla': 'r',
        'delta_chla': '#287D8E',
    }
    line_yticks_dic = {
        '2020_temp': [0, 10, 20, 30 , 40],
        '2100_temp': [0, 10, 20, 30 , 40],
        'delta_temp': [2, 4, 6, 8, 10],
        '2020_chla': [0, 10, 20, 30 , 40],
        '2100_chla': [0, 10, 20, 30 , 40],
        'delta_chla': [2, 4, 6, 8, 10],
    }
    label_dic = {
        '2020_temp': 'Temperature (2020)',
        '2100_temp': 'Temperature (2100)',
        'delta_temp': 'ΔTemperature',
        '2020_chla': 'Chla (2020)',
        '2100_chla': 'Chla (2100)',
        'delta_chla': 'ΔChla',
    }
    line_ylabel_dic = {
        '2020_temp': 'Growing-season temperature (°C)',
        '2100_temp': 'Growing-season temperature (°C)',
        'delta_temp': 'ΔTemperature (°C)',
        '2020_chla': 'Growing-season Chla (µg/L)',
        '2100_chla': 'Growing-season Chla (µg/L)',
        'delta_chla': 'ΔChla (µg/L)',
    }
    line_label_dic = {
        '2020_temp': '2020',
        '2100_temp': '2100 (SSP5-8.5)',
        'delta_temp': '2100 - 2020',
        '2020_chla': '2020',
        '2100_chla': '2100 (SSP5-8.5)',
        'delta_chla': '2100 - 2020',
    }

    for i_c, c in enumerate(cols):
        for i_r, r in enumerate(years + ['delta']):
            ax = axes[i_r, i_c]
            m = Basemap(projection='cyl', resolution='i', ax=ax)
            m.drawcountries(linewidth=0.1)
            m.fillcontinents(color='lightgray', lake_color='lightgray', alpha=.5)
            m.drawmapboundary(fill_color='none', linewidth=0.5)
            parallels = np.arange(-80, 81, 40)
            meridians = np.arange(-180, 181, 60)
            m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25, dashes=[5, 0], fontsize=6, color='lightgray')
            m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25, dashes=[5, 0], fontsize=6, color='lightgray')
            ax.set_title(title_dic['{}_{}'.format(r, c)], fontsize=8)

            bar_ticks = car_ticks_dic['{}_{}'.format('delta' if r == 'delta' else 'year', c)]
            bar_ticklabels = [str(t) for t in bar_ticks]
            bar_ticklabels[-1] = '>' + bar_ticklabels[-1]

            values = filter_data_df['{}_{}'.format(r, c)]
            norm = mpl.colors.Normalize(vmin=bar_ticks[0], vmax=bar_ticks[-1], clip=True)
            x, y = m(filter_data_df['Pour_long'].values, filter_data_df['Pour_lat'].values)
            scatter = m.scatter(
                x, y,
                c=values,
                cmap='viridis',
                norm=norm,
                marker='s',
                s=.1,
                edgecolor='none',
                alpha=1,
            )
            ax.set_aspect('auto')
            cb = m.colorbar(scatter, location='right', size="2%",pad='2%')
            cb.set_label(cb_dic['{}_{}'.format(r, c)], fontsize=6)
            cb.ax.tick_params(which='both', direction='out', width=0.5, length=1, labelsize=6)
            cb.set_ticks(bar_ticks)
            cb.set_ticklabels(bar_ticklabels, fontsize=6)

            ax_inner = ax.inset_axes([0.07, 0.12, 0.2, 0.3], zorder=100)
            sns.kdeplot(values, fill=True, color=line_c, alpha=1, linewidth=2, ax=ax_inner)
            ax_inner.set_xlabel("")
            ax_inner.set_ylabel('Density', fontsize=6)
            ax_inner.set_yticks([])
            kde_xticks = kde_xticks_dic['{}_{}'.format('delta' if r == 'delta' else 'year', c)]
            ax_inner.set_xticks(kde_xticks)
            ax_inner.tick_params(axis='both', which='major', width=.5, length=1, labelsize=6)
            for spine in ax_inner.spines.values():
                spine.set_linewidth(.25)

        ax = axes[3, i_c]
        plt.sca(ax)
        width = 1 if c == 'temp' else 2
        lat_bins = np.arange(-90, 91, width)
        filter_data_df['lat_bin'] = pd.cut(filter_data_df['Pour_lat'], bins=lat_bins, include_lowest=True)
        for i_r, r in enumerate(years):
            grouped = filter_data_df.groupby('lat_bin', as_index=False).agg({
                '{}_{}'.format(r, c): 'mean',
                'Pour_lat': 'mean'
            })
            ax.plot(grouped['Pour_lat'], grouped['{}_{}'.format(r, c)], lw=1.5, c=color_dic['{}_{}'.format(r, c)], alpha=1, label=line_label_dic['{}_{}'.format(r, c)])
        plt.xticks(range(-80, 81, 40), ['80°S', '40°S', '0°', '40°N', '80°N'])
        plt.xlim(-80, 80)
        yticks = line_yticks_dic['2020_{}'.format(c)]
        plt.yticks(yticks)
        plt.ylim(yticks[0], yticks[-1])
        ax.set_xlabel('Latitude', fontsize=7)
        ax.set_ylabel(line_ylabel_dic['2020_{}'.format(c)], fontsize=7)
        ax.tick_params(axis='both', which='major', width=.5, length=2, labelsize=6)
        plt.legend(loc='upper left', fontsize=6)

        ax_tx = ax.twinx()
        plt.sca(ax_tx)
        for i_r, r in enumerate(['delta']):
            grouped = filter_data_df.groupby('lat_bin', as_index=False).agg({
                '{}_{}'.format(r, c): 'mean',
                'Pour_lat': 'mean'
            })
            ax_tx.plot(grouped['Pour_lat'], grouped['{}_{}'.format(r, c)], lw=1.5, c=color_dic['{}_{}'.format(r, c)], alpha=1, label=line_label_dic['{}_{}'.format(r, c)])
        yticks = line_yticks_dic['delta_{}'.format(c)]
        plt.yticks(yticks)
        plt.ylim(yticks[0], yticks[-1])
        ax_tx.set_ylabel(line_ylabel_dic['delta_{}'.format(c)], fontsize=7)
        ax_tx.tick_params(axis='both', which='major', width=.5, length=2, labelsize=6)
        plt.legend(loc='upper right', fontsize=6)

    text_dic = {axes[0, 0]: 'a', axes[0, 1]: 'b', axes[1, 0]: 'c', axes[1, 1]: 'd', axes[2, 0]: 'e', axes[2, 1]: 'f', axes[3, 0]: 'g', axes[3, 1]: 'h'}
    for ax, text in text_dic.items():
        ax.text(0.05, 0.7, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('Figs/FS Global lake future temp chla map.png', dpi=300)


def fs_data_relation():
    data_dic = {}
    data_df = get_lake_wq(if_china=True)
    print('filter has all data records', len(data_df))
    print('filter has all data lakes', len(data_df.groupby('Unique Lake').mean(numeric_only=True)))

    meta_df = get_lake_meta()
    for c in meta_df.columns:
        if c not in data_df.columns:
            data_df[c] = data_df['Unique Lake'].map(meta_df[c])

    data_df = data_df.dropna(subset=['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'])
    data_dic['0: Raw data'] = data_df.copy()

    data_df = data_df.dropna(subset=['sampledate', 'Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'])
    data_dic['1: Records with complete\nChla, TN, TP, and sampling date'] = data_df.copy()

    data_df = data_df[((data_df['Chla (µg/L)'] < 200) & (data_df['Chla (µg/L)'] > 1)) | (data_df['Chla (µg/L)'].isna())]
    data_df = data_df[((data_df['TP (µg/L)'] < 200) & (data_df['TP (µg/L)'] > 1)) | (data_df['TP (µg/L)'].isna())]
    data_dic['2: Records with 1 < Chla < 200 μg/L\nand 1 < TP < 200 μg/L'] = data_df.copy()

    data_df = filter_summer_data(data_df)
    data_df = data_df[data_df['year'] >= 1980]
    data_dic['3: Records within growing season'] = data_df.copy()

    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=False, if_filter=True)
    used_data_df['Unique Lake'] = used_data_df['Unique Lake'].astype(str)
    data_dic['4: Lakes with robust long-term data'] = used_data_df.copy()

    fig, axes = plt.subplots(5, 2, figsize=(8, 11))
    for i, (title, df) in enumerate(data_dic.items()):
        china_df = df[df['Unique Lake'].str.startswith('CHNL', na=False)]
        global_df = df[~df['Unique Lake'].str.startswith('CHNL', na=False)]
        china_mean_df = china_df.groupby('Unique Lake').mean(numeric_only=True)
        global_mean_df = global_df.groupby('Unique Lake').mean(numeric_only=True)
        print(title, len(china_df), len(global_df), df.columns)

        plt.sca(axes[i, 0])
        m = Basemap(projection='cyl', resolution='i', ax=axes[i, 0])
        m.drawcountries(linewidth=0.1)
        m.fillcontinents(color='lightgray', lake_color='lightgray', alpha=.5)
        m.drawmapboundary(fill_color='none', linewidth=0.5)
        parallels = np.arange(-80, 81, 40)
        meridians = np.arange(-180, 181, 60)
        m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')
        m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')
        m.scatter(*m(global_mean_df['Longitude'].values, global_mean_df['Latitude'].values), marker='o', s=2, c=scat_c, edgecolors=scat_c, linewidths=0.5, alpha=.5)
        m.scatter(*m(china_mean_df['Longitude'].values, china_mean_df['Latitude'].values), marker='o', s=2, c='g', edgecolors='g', linewidths=0.5, alpha=.5)
        axes[i, 0].set_aspect('auto')
        plt.text(0.02, 0.95, title + ' (Lakes)', fontsize=8, ha='left', va='top', color='k', transform=axes[i, 0].transAxes)

        x_col = 'TP (µg/L)'
        plt.sca(axes[i, 1])
        plt.scatter(global_df[x_col], global_df['Chla (µg/L)'], alpha=.3, c=scat_c, edgecolors=scat_c, s=5, linewidths=.5, label='Global dataset')
        plt.scatter(china_df[x_col], china_df['Chla (µg/L)'], alpha=.3, c='g', edgecolors='g', s=5, linewidths=.5, label="China's dataset")
        lowess_result = sm.nonparametric.lowess(df['Chla (µg/L)'], df[x_col], frac=.3)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100, label='LOWESS')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(x_col, fontsize=8)
        plt.ylabel('Chla (µg/L)', fontsize=8)
        plt.text(0.02, 0.95, title + ' (Records)', fontsize=8, ha='left', va='top', color='k', transform=axes[i, 1].transAxes)
        if i==0:
            plt.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig('Figs/FS Lake water quality relation.png', dpi=300)


def fs_data_reg_compare():
    data_dic = {}
    data_df = get_lake_wq(if_china=True)
    print('filter has all data records', len(data_df))
    print('filter has all data lakes', len(data_df.groupby('Unique Lake').mean(numeric_only=True)))

    meta_df = get_lake_meta()
    for c in meta_df.columns:
        if c not in data_df.columns:
            data_df[c] = data_df['Unique Lake'].map(meta_df[c])

    data_df = data_df.dropna(subset=['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'])
    data_dic['0: Raw data'] = data_df.groupby('Unique Lake').mean(numeric_only=True)

    data_df = data_df.dropna(subset=['sampledate', 'Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'])
    data_dic['1: Records with complete\nChla, TN, TP, and sampling date'] = data_df.groupby('Unique Lake').mean(numeric_only=True)

    data_df = data_df[((data_df['Chla (µg/L)'] < 200) & (data_df['Chla (µg/L)'] > 1)) | (data_df['Chla (µg/L)'].isna())]
    data_df = data_df[((data_df['TP (µg/L)'] < 200) & (data_df['TP (µg/L)'] > 1)) | (data_df['TP (µg/L)'].isna())]
    data_dic['2: Records with 1 < Chla < 200 μg/L\nand 1 < TP < 200 μg/L'] = data_df.groupby('Unique Lake').mean(numeric_only=True)

    data_df = filter_summer_data(data_df)
    data_df = data_df[data_df['year'] >= 1980]
    data_dic['3: Records within growing season'] = data_df.groupby('Unique Lake').mean(numeric_only=True)

    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=True, if_filter=False)
    print('filter n lake', len(used_data_df))
    used_data_df = filter_lakes_with_rse(used_data_df, t_chl=0.5)
    print('filter n lake with rse', len(used_data_df))
    data_dic['4: Lakes with robust long-term data'] = used_data_df.copy()

    fig, axes = plt.subplots(5, 2, figsize=(7, 12))
    for i, (title, df) in enumerate(data_dic.items()):
        china_df = df[df.index.str.startswith('CHNL', na=False)]
        global_df = df[~df.index.str.startswith('CHNL', na=False)]
        print(title, len(china_df), len(global_df), df.columns)
        for j, x_col in enumerate(['TP (µg/L)', 'TN (µg/L)']):
            plt.sca(axes[i, j])

            plt.scatter(global_df[x_col], global_df['Chla (µg/L)'], alpha=.3, c=scat_c, edgecolors=scat_c, s=5, linewidths=.5)
            plt.scatter(china_df[x_col], china_df['Chla (µg/L)'], alpha=.3, c='g', edgecolors='g', s=5, linewidths=.5)
            slope, intercept, r_value, p_value, std_err = linregress(df[x_col], df['Chla (µg/L)'])
            plt.plot(df[x_col], df[x_col] * slope + intercept, line_c, linewidth=2, zorder=100, label='OLS regression (R$^2$={:.2f}, n={})'.format(r_value**2, len(df[x_col])))
            plt.xlabel('Lake mean ' + x_col, fontsize=8)
            plt.ylabel('Lake mean Chla (µg/L)', fontsize=8)
            plt.text(0.02, 0.95, title, fontsize=8, ha='left', va='top', color='k', transform=axes[i, j].transAxes)
            plt.legend(fontsize=8, loc='center left', frameon=False)
    plt.tight_layout()
    plt.savefig('Figs/FS Lake reg compare.png', dpi=300)


def fs_lake_dist_along_depth_alt():
    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=True, if_filter=False)
    used_data_df = filter_lakes_with_rse(used_data_df, t_chl=0.5)

    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Average depth (m)'] = used_data_df['depth']
    used_data_df['Absolute latitude (°)'] = np.abs(used_data_df['Latitude'])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for i, x_col in enumerate(['Average depth (m)', 'Absolute latitude (°)']):
        plt.sca(axes[i])
        plt.hist(used_data_df[x_col], bins=30, edgecolor='k', alpha=0.6, color=scat_c)
        plt.xlabel(x_col)
        if i == 0:
            plt.ylabel('Number of lakes')
    plt.tight_layout()
    plt.savefig('Figs/FS used lake dist.png', dpi=300)


def fs_surface():
    used_data_df = filter_lakes(n_year=5, t_depth=100, if_mean=True, if_filter=False)
    used_data_df = filter_lakes_with_rse(used_data_df, t_chl=0.5)

    used_data_df['Absolute latitude'] = np.abs(used_data_df['Latitude'])
    used_data_df['TN:TP'] = used_data_df['TN (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Chla:TP'] = used_data_df['Chla (µg/L)'] / used_data_df['TP (µg/L)']
    used_data_df['Average depth (m)'] = used_data_df['depth']
    used_data_df['LN (µg P/L)'] = used_data_df['ln']

    used_data_df['Long-term growing-season Chla (µg/L)'] = used_data_df['Chla (µg/L)']
    used_data_df['Long-term growing-season LN (µg P/L)'] = used_data_df['LN (µg P/L)']
    used_data_df['Long-term growing-season Chla / LN \n(µg Chla / µg P)'] = np.log10(used_data_df['Long-term growing-season Chla (µg/L)']) / np.log10(used_data_df['Long-term growing-season LN (µg P/L)'])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    cs = ['Long-term growing-season LN (µg P/L)', 'Long-term growing-season Chla (µg/L)', 'Long-term growing-season Chla / LN \n(µg Chla / µg P)']
    for i_c, c in enumerate(cs):
        plt.sca(axes[i_c])
        x, y, z = used_data_df['Average depth (m)'], used_data_df['Absolute latitude'], used_data_df[c]
        X = np.column_stack([x, y, x * y])
        model = LinearRegression().fit(X, z)
        xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        ZZ = model.predict(np.column_stack([xx.ravel(), yy.ravel(), xx.ravel() * yy.ravel()]))
        ZZ = ZZ.reshape(xx.shape)
        plt.contourf(xx, yy, ZZ, levels=30, cmap='viridis')
        sc = plt.scatter(x, y, c=z, cmap='viridis', s=30, linewidths=.5, edgecolor='k')
        cbar = plt.colorbar(sc)
        cbar.set_label(c, fontsize=9)
        cbar.ax.tick_params(labelsize=8, width=.5, length=2)
        cbar.outline.set_linewidth(.5)
        if i_c == 0:
            plt.ylabel("Absolute latitude (°)", fontsize=9)
        if i_c == 1:
            plt.xlabel('Average depth (m)', fontsize=9)
    plt.tight_layout()
    plt.savefig('Figs/FS surface.png', dpi=300)


def fs_shallow_lake_responses():
    used_data_df = filter_lakes()
    used_data_df['Unique Lake'] = used_data_df.index
    lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
    lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
    used_lake_ids = used_data_df.index.intersection(lake_climate_df.index)
    used_lake_cli_df = lake_climate_df.loc[used_lake_ids]
    merge_df = pd.concat([used_data_df, used_lake_cli_df], axis=1)

    abb_dic = {
        'Lat': 'Latitude',
        'Temp': 'temperature_2m',
        'SR': 'surface_solar_radiation_downwards_sum',
        'WS': 'wind_10m',
        'Pr': 'total_precipitation_sum',
        'TN': 'TN (µg/L)',
        'TP': 'TP (µg/L)',
        'LN': 'ln',
        'Chla': 'Chla (µg/L)',
    }
    for k, v in abb_dic.items():
        merge_df[k] = merge_df[v]

    plt.figure(figsize=(6, 6))
    windows = [[20, 40], [30, 45], [40, 55], [42.5, 60], [45, 65]]
    cs = create_color_dict(range(len(windows)), 'Reds_r')
    norm = mpl.colors.Normalize(vmin=np.percentile(merge_df['Latitude'], 2.5), vmax=np.percentile(merge_df['Latitude'], 97.5), clip=True)
    plt.scatter(merge_df['ln'], merge_df['Chla (µg/L)'], c=merge_df['Latitude'], cmap='Reds_r', norm=norm, edgecolors='k', linewidths=.2, label='Shallow lakes (n=398)\nColors represent Lat. gradient (Low [Red] to High [White])')
    for i, (lat_min, lat_max) in enumerate(windows):
        sub_df = used_data_df[used_data_df['Latitude'] > lat_min][used_data_df['Latitude'] < lat_max]
        x, y, lat = sub_df['ln'], sub_df['Chla (µg/L)'], sub_df['Latitude']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        plt.plot(x, x * slope + intercept, c=cs[i], label='Lakes within {}°-{}° (Slope={:.2f})'.format(int(lat_min), int(lat_max), slope))
    plt.xlabel("Long-term growing-season {}".format('LN (µg P/L)'), fontsize=9)
    plt.ylabel("Long-term growing-season {}".format('Chla (µg/L)'), fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figs/FS shallow lake responses.png', dpi=300)


def fs_bayesian_model_data_dist():
    used_data_df = filter_lakes()
    used_data_df['Unique Lake'] = used_data_df.index
    lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
    lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
    used_lake_ids = used_data_df.index.intersection(lake_climate_df.index)
    used_lake_cli_df = lake_climate_df.loc[used_lake_ids]
    merge_df = pd.concat([used_data_df, used_lake_cli_df], axis=1)

    abb_dic = {
        'Lat': 'Latitude',
        'Temp': 'temperature_2m',
        'SR': 'surface_solar_radiation_downwards_sum',
        'WS': 'wind_10m',
        'Pr': 'total_precipitation_sum',
        'TN': 'TN (µg/L)',
        'TP': 'TP (µg/L)',
        'LN': 'ln',
        'Chla': 'Chla (µg/L)',
    }
    for k, v in abb_dic.items():
        merge_df[k] = merge_df[v]
    cols = list(abb_dic.keys())

    def draw_lowess(x, y, frac=0.5, color=line_c, lw=1.0, **kwargs):
        smoothed = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color=color, lw=lw)

    g = sns.PairGrid(merge_df[cols], corner=True, diag_sharey=False)
    g.map_lower(
        sns.scatterplot,
        s=5,
        alpha=0.3,
        color=scat_c,
        edgecolor=scat_c
    )
    g.map_lower(
        draw_lowess, color=line_c
    )
    g.map_diag(
        sns.kdeplot,
        fill=True,
        color=line_c,
        alpha=1,
        linewidth=1
    )
    max_vals = []
    for col in cols:
        from scipy.stats import gaussian_kde
        data = merge_df[col].dropna().values
        kde = gaussian_kde(data)
        x_eval = np.linspace(data.min(), data.max(), 1000)
        y_eval = kde(x_eval)
        max_vals.append(y_eval.max())
    for ax, ymax in zip(g.diag_axes, max_vals):
        ax.set_ylim(0, ymax * 1.25)
    for ax in g.axes.flatten():
        if ax is not None:
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
            ax.tick_params(
                axis='both',
                which='major',
                width=.5,
                length=3,
                labelsize=8,
                direction='in'
            )
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)
    g.fig.subplots_adjust(wspace=0.1, hspace=0.1)
    g.fig.set_size_inches(8, 6)
    plt.tight_layout()
    plt.savefig('Figs/FS model pair plot.png', dpi=300)


def fs_global_lake_chla_available():
    data_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col='Hylak_id')
    chla_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_mean_bayes.csv', index_col='Hylak_id')
    data_df = pd.concat([data_df, chla_df], axis=1)
    shallow_data_df = data_df[data_df['depth'] < 6]
    filter_data_df = data_df.dropna(subset=['ln', '2020'])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    ax2, ax1 = axes
    plt.sca(ax2)
    width = 2
    lat_bins = np.arange(-90, 91, width)
    lat_labels = lat_bins[:-1]
    shallow_counts, _ = np.histogram(shallow_data_df['Pour_lat'], bins=lat_bins)
    chla_counts, _ = np.histogram(filter_data_df['Pour_lat'], bins=lat_bins)
    ax2.bar(lat_labels, shallow_counts, color='#B0C4DE', edgecolor='none', alpha=0.8, width=width * .75,
            label='All shallow lakes (n={})'.format(len(shallow_data_df)))
    ax2.bar(lat_labels, chla_counts, color=scat_c, edgecolor='none', alpha=1, width=width * .75,
            label='Shallow lakes with derived Chla (n={})'.format(len(filter_data_df)))
    plt.xticks(range(-80, 81, 20), ['80°S', '60°S', '40°S', '20°S', '0°', '20°N', '40°N', '60°N', '80°N'])
    plt.xlim(-80, 80)
    ax2.set_yticks([0, 2e4, 4e4, 6e4, 8e4, 10e4, 12e4], [0, 2, 4, 6, 8, 10, 12])
    ax2.set_ylim([0, 12e4])
    ax2.set_ylabel('Number of lakes (×10$^4$)')
    ax2.set_xlabel('Latitude')
    plt.legend(loc='upper left')
    ax2_tx = ax2.twinx()
    ax2_tx.plot(lat_labels, chla_counts / shallow_counts, lw=2, c=line_c, alpha=1, label='Proportion of lakes with derived Chla')
    ax2_tx.set_ylim([0, 1.2])
    ax2_tx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], [0, 20, 40, 60, 80, 100, 120])
    ax2_tx.set_ylabel('Proportion of lakes with derived Chla (%)')
    plt.legend(loc='upper right')

    plt.sca(ax1)
    width = 1
    area_bins = [0.1, 0.2, 0.5, 1, 10, 100, 10000, 1000000]
    shallow_counts, _ = np.histogram(shallow_data_df['Lake_area'], bins=area_bins)
    chla_counts, _ = np.histogram(filter_data_df['Lake_area'], bins=area_bins)
    area_labels = ['{}-{}\n(n={})'.format(area_bins[i], area_bins[i+1], shallow_counts[i]) for i in range(len(area_bins)-1)]
    ax1.bar(range(len(area_bins)-1), shallow_counts, color='#B0C4DE', edgecolor='none', alpha=0.8, width=width * .75,
            label='All shallow lakes (n={})'.format(len(shallow_data_df)))
    ax1.bar(range(len(area_bins)-1), chla_counts, color=scat_c, edgecolor='none', alpha=1, width=width * .75,
            label='Shallow lakes with derived Chla (n={})'.format(len(filter_data_df)))
    plt.xticks(range(len(area_bins)-1), area_labels)
    ax1.set_yticks([0, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5], [0, 1, 2, 3, 4, 5, 6])
    ax1.set_ylim([0, 6e5])
    ax1.set_ylabel('Number of lakes (×10$^5$)')
    ax1.set_xlabel('Lake area (km$^2$)')
    plt.legend(loc='upper left')

    ax1_tx = ax1.twinx()
    ax1_tx.plot(range(len(area_bins)-1), chla_counts / shallow_counts, lw=2, c=line_c, alpha=1,
                label='Proportion of lakes with derived Chla')
    ax1_tx.set_ylim([0, 1.2])
    ax1_tx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], [0, 20, 40, 60, 80, 100, 120])
    ax1_tx.set_ylabel('Proportion of lakes with derived Chla (%)')
    plt.legend(loc='upper right')
    text_dic = {ax2: 'a', ax1: 'b'}
    for ax, text in text_dic.items():
        ax.text(0.05, 0.8 if text == 'a' else 0.80, text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='top', ha='left', zorder=100)
    plt.tight_layout()
    plt.savefig('Figs/FS Global lake chla available.png', dpi=300)


def fs_chla_retrievals_and_mea_compare():
    used_data_df = filter_lakes(n_year=1, t_depth=100, if_mean=True, if_filter=False)

    hylak_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col=0)
    coords_all = np.column_stack((hylak_df['Pour_long'].values, hylak_df['Pour_lat'].values))
    coords_local = np.column_stack((used_data_df['Longitude'].values, used_data_df['Latitude'].values))
    from scipy.spatial import cKDTree
    tree = cKDTree(coords_all)
    dist, idx = tree.query(coords_local, k=1)
    used_data_df['Hylak_id'] = hylak_df.index.values[idx]
    used_data_df = used_data_df.join(hylak_df, on='Hylak_id', rsuffix='_hy')

    used_data_df = used_data_df.dropna(subset=['Chla (µg/L)', 'Chla (µg/L)_hy'])
    used_data_df = used_data_df[used_data_df['Chla (µg/L)'] < 200]
    used_data_df = used_data_df[used_data_df['Chla (µg/L)_hy'] < 200]

    plt.figure(figsize=(8, 6))
    r2 = r2_score(used_data_df['Chla (µg/L)'], used_data_df['Chla (µg/L)_hy'])
    plt.scatter(used_data_df['Chla (µg/L)'], used_data_df['Chla (µg/L)_hy'], c=scat_c, edgecolors='k', s=30, linewidths=.5, alpha=.8, label='R$^2$={:.2f}, n={}'.format(r2, len(used_data_df)))
    plt.plot([0, 80], [0, 80], '-', lw=0.5, alpha=1, color='k', label="1:1 line")
    print('filter n lake compare', len(used_data_df), r2)
    plt.xlabel("Measured long-term growing-season Chla (µg/L)")
    plt.ylabel("Satellite-derived growing-season Chla (µg/L)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('Figs/FS chla retrievals and mea compare.png', dpi=300)
    plt.show()


def fs_ln_retrievals_and_mea_compare():
    used_data_df = filter_lakes(n_year=1, t_depth=100, if_mean=True, if_filter=False)

    hylak_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col=0)
    print('hylak records', len(hylak_df), hylak_df.columns)
    coords_all = np.column_stack((hylak_df['Pour_long'].values, hylak_df['Pour_lat'].values))
    coords_local = np.column_stack((used_data_df['Longitude'].values, used_data_df['Latitude'].values))
    from scipy.spatial import cKDTree
    tree = cKDTree(coords_all)
    dist, idx = tree.query(coords_local, k=1)
    used_data_df['Hylak_id'] = hylak_df.index.values[idx]
    used_data_df = used_data_df.join(hylak_df, on='Hylak_id', rsuffix='_hy')
    hylak_df = hylak_df.dropna(subset=['ln'])

    used_data_df = used_data_df.dropna(subset=['Chla (µg/L)', 'Chla (µg/L)_hy'])
    used_data_df = used_data_df[used_data_df['Chla (µg/L)'] < 200]
    used_data_df = used_data_df[used_data_df['Chla (µg/L)_hy'] < 200]
    plt.figure(figsize=(8, 4))
    all_data_df = filter_lakes(n_year=1, t_depth=100, if_mean=False, if_filter=False)
    for k, (v, c, h, bw) in {'Derived LN': (hylak_df['ln'].values, line_c, 0.8, 3), 'Measured LN': (all_data_df['ln'].values, scat_c, 0.7, 4)}.items():
        v = np.log10(v[v>1])
        sns.kdeplot(v, fill=True, color=c, linewidth=1, alpha=0.4, label=k, bw_adjust=bw)
        plt.axvline(np.nanmedian(v), color=c, linestyle='--',linewidth=2)
        plt.text(np.nanmedian(v), plt.ylim()[1] * h, f'Median = {np.power(10, np.nanmedian(v)):.1f} µg P/L', ha='center', color=c, fontsize=8)
    plt.legend(loc='upper right')
    plt.xlabel("LN (log µg P/L)")
    plt.tight_layout()
    plt.savefig('Figs/FS LN retrievals and mea compare.png', dpi=300)


def fs_different_longterm_sample_differ():
    n_years = [1, 3, 5 ,6, 7]
    text_years = [1, 3, 5 ,7, 9]
    N = 250

    fig, axes = plt.subplots(len(n_years), 2, figsize=(8, 10), sharex='col')
    for i_y, n_year in enumerate(n_years):
        mean_data_df = filter_lakes(n_year=n_year, t_depth=25, if_mean=True, if_filter=False)
        lat_mini, length, step, n, n_sam = -4, 5, .25, 80, N
        lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
        xs, ys, rs, fs, ics, gis, gls, gips, ns, ats, wts = [], [], [], [], [], [], [], [], [], [], []
        for lat_min, lat_max in lat_windows:
            for i in range(n_sam):
                samples = sample_lakes_in_depth_band(mean_data_df, lat_min, lat_max, n=30, ln_bins=10, random_state=911 + i)
                if len(samples) >= 10:
                    x, y = samples['ln'], samples['Chla (µg/L)']
                    x_with_const = sm.add_constant(x)
                    glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                    y_pred = glm_model.predict(x_with_const)
                    slope, p_value = glm_model.params.loc['ln'], glm_model.pvalues.loc['ln']
                    if p_value < 0.05:
                        fs.append(r2_score(y, y_pred))
                        ys.append(slope)
                        xs.append(samples['depth'].mean())

        plt.sca(axes[i_y, 0])
        xs = np.array(xs)
        ys = np.array(ys)
        plt.scatter(xs[xs < 6], ys[xs < 6], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        plt.scatter(xs[xs >= 6], ys[xs >= 6], alpha=.25, c=depth_color, edgecolors=depth_color, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(ys, xs, frac=.5)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)

        plt.xlim([0, 20])
        plt.xticks([0, 5, 10, 15, 20])
        plt.ylim([.2, .8])
        plt.yticks([.2, .4, .6, .8])
        if i_y == len(n_years) - 1:
            plt.xlabel('Lake average depth (m)', fontsize=9)
        if i_y == 2:
            plt.ylabel('Nutrient-Chl$a$ slope (µg Chl$a$ / µg P)', fontsize=9)
        axes[i_y, 0].text(0.05, 0.95, "Minimum record length = {} years".format(text_years[i_y]), transform=axes[i_y, 0].transAxes, fontsize=9, va='top', ha='left')


        used_data_df = filter_lakes(n_year=n_year, t_depth=6)
        used_data_df['Unique Lake'] = used_data_df.index
        valid_lakes = used_data_df.index
        lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
        lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
        cli_lake_ids = [valid_lake for valid_lake in valid_lakes if valid_lake in lake_climate_df.index]
        used_lake_cli_df = lake_climate_df.loc[cli_lake_ids]

        lat_mini, length, step, n, n_sam = 15, 15, .5, 70, N
        lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
        xs, ys, fs, ics, ats, wts = [], [], [], [], [], []
        for lat_min, lat_max in lat_windows:
            for i in range(n_sam):
                samples = sample_lakes_in_lat_band(used_data_df, lat_min, lat_max, n=30, tp_bins=10,
                                                   random_state=911 + i + int(lat_min))
                if len(samples) >= 10:
                    x, y = samples['ln'], samples['Chla (µg/L)']
                    x_with_const = sm.add_constant(x)
                    glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                    y_pred = glm_model.predict(x_with_const)
                    slope, intercept, p_value = glm_model.params.loc['ln'], glm_model.params.loc['const'], \
                        glm_model.pvalues.loc['ln']
                    if p_value < 0.05:
                        xs.append(samples['Latitude'].mean())
                        ys.append(slope)
                        fs.append(r2_score(y, y_pred))
                        ics.append(intercept)
                        ats.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'temperature_2m']))
                        wts.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'lake_mix_layer_temperature']))

        plt.sca(axes[i_y, 1])
        plt.scatter(xs, ys, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(ys, xs, frac=.125)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)
        plt.xlim([25, 55])
        plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
        plt.ylim([0, 1.5])
        plt.yticks([0, .5, 1, 1.5])
        if i_y == len(n_years) - 1:
            plt.xlabel('Absolute latitude', fontsize=9)
        if i_y == 2:
            plt.ylabel('Nutrient-Chl$a$ slope in shallow lakes\n(µg Chl$a$ / µg P)', fontsize=9)

    plt.tight_layout()
    plt.savefig('Figs/FS sensitivity of long-term filter.png', dpi=300)


def fs_different_rse_sample_differ():
    rse_ls = [30, 50, 70]
    N = 250

    fig, axes = plt.subplots(len(rse_ls), 2, figsize=(8, 6), sharex='col')
    for i_y, rse in enumerate(rse_ls):
        mean_data_df = filter_lakes(n_year=5, t_depth=25, if_mean=True, if_filter=False)
        lat_mini, length, step, n, n_sam = -4, 5, .25, 80, N
        lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
        xs, ys, rs, fs, ics, gis, gls, gips, ns, ats, wts = [], [], [], [], [], [], [], [], [], [], []
        for lat_min, lat_max in lat_windows:
            for i in range(n_sam):
                samples = sample_lakes_in_depth_band(mean_data_df, lat_min, lat_max, n=30, ln_bins=10,
                                                     random_state=911 + i)
                if len(samples) >= 10:
                    x, y = samples['ln'], samples['Chla (µg/L)']
                    x_with_const = sm.add_constant(x)
                    glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                    y_pred = glm_model.predict(x_with_const)
                    slope, p_value = glm_model.params.loc['ln'], glm_model.pvalues.loc['ln']
                    if p_value < 0.05:
                        fs.append(r2_score(y, y_pred))
                        ys.append(slope)
                        xs.append(samples['depth'].mean())
        plt.sca(axes[i_y, 0])
        xs = np.array(xs)
        ys = np.array(ys)
        plt.scatter(xs[xs < 6], ys[xs < 6], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        plt.scatter(xs[xs >= 6], ys[xs >= 6], alpha=.25, c=depth_color, edgecolors=depth_color, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(ys, xs, frac=.5)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)

        plt.xlim([0, 20])
        plt.xticks([0, 5, 10, 15, 20])
        plt.ylim([.2, .8])
        plt.yticks([.2, .4, .6, .8])
        if i_y == len(rse_ls) - 1:
            plt.xlabel('Lake average depth (m)', fontsize=9)
        if i_y == 1:
            plt.ylabel('Nutrient-Chl$a$ slope (µg Chl$a$ / µg P)', fontsize=9)
        axes[i_y, 0].text(0.05, 0.95, "RSE ≤ {}%".format(rse),
                          transform=axes[i_y, 0].transAxes, fontsize=9, va='top', ha='left')

        used_data_df = filter_lakes(n_year=5, t_depth=6)
        used_data_df['Unique Lake'] = used_data_df.index
        valid_lakes = used_data_df.index
        lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'),
                                      header=0, index_col=0)
        lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
        cli_lake_ids = [valid_lake for valid_lake in valid_lakes if valid_lake in lake_climate_df.index]
        used_lake_cli_df = lake_climate_df.loc[cli_lake_ids]

        lat_mini, length, step, n, n_sam = 15, 15, .5, 70, N
        lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
        xs, ys, fs, ics, ats, wts = [], [], [], [], [], []
        for lat_min, lat_max in lat_windows:
            for i in range(n_sam):
                samples = sample_lakes_in_lat_band(used_data_df, lat_min, lat_max, n=30, tp_bins=10,
                                                   random_state=911 + i + int(lat_min))
                if len(samples) >= 10:
                    x, y = samples['ln'], samples['Chla (µg/L)']
                    x_with_const = sm.add_constant(x)
                    glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                    y_pred = glm_model.predict(x_with_const)
                    slope, intercept, p_value = glm_model.params.loc['ln'], glm_model.params.loc['const'], \
                        glm_model.pvalues.loc['ln']
                    if p_value < 0.05:
                        xs.append(samples['Latitude'].mean())
                        ys.append(slope)
                        fs.append(r2_score(y, y_pred))
                        ics.append(intercept)
                        ats.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'temperature_2m']))

        plt.sca(axes[i_y, 1])
        plt.scatter(xs, ys, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(ys, xs, frac=.125)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)
        plt.xlim([25, 55])
        plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
        plt.ylim([0, 1.5])
        plt.yticks([0, .5, 1, 1.5])
        if i_y == len(rse_ls) - 1:
            plt.xlabel('Absolute latitude', fontsize=9)
        if i_y == 1:
            plt.ylabel('Nutrient-Chl$a$ slope in shallow lakes\n(µg Chl$a$ / µg P)', fontsize=9)

    plt.tight_layout()
    plt.savefig('Figs/FS sensitivity of rse filter.png', dpi=300)


def fs_different_depth_sample_differ():
    depth_ls = [4, 5, 6, 7, 8]
    N = 250

    fig, axes = plt.subplots(len(depth_ls), 2, figsize=(8, 10), sharex='col')
    for i_y, depth in enumerate(depth_ls):
        used_data_df = filter_lakes(n_year=5, t_depth=depth)
        used_data_df['Unique Lake'] = used_data_df.index
        valid_lakes = used_data_df.index
        lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'),
                                      header=0, index_col=0)
        lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
        cli_lake_ids = [valid_lake for valid_lake in valid_lakes if valid_lake in lake_climate_df.index]
        used_lake_cli_df = lake_climate_df.loc[cli_lake_ids]

        lat_mini, length, step, n, n_sam = 15, 15, .5, 70, N
        lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
        xs, ys, fs, ics, ats, wts = [], [], [], [], [], []
        for lat_min, lat_max in lat_windows:
            for i in range(n_sam):
                samples = sample_lakes_in_lat_band(used_data_df, lat_min, lat_max, n=30, tp_bins=10,
                                                   random_state=911 + i + int(lat_min))
                if len(samples) >= 10:
                    x, y = samples['ln'], samples['Chla (µg/L)']
                    x_with_const = sm.add_constant(x)
                    glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                    y_pred = glm_model.predict(x_with_const)
                    slope, intercept, p_value = glm_model.params.loc['ln'], glm_model.params.loc['const'], \
                        glm_model.pvalues.loc['ln']
                    if p_value < 0.05:
                        xs.append(samples['Latitude'].mean())
                        ys.append(slope)
                        fs.append(r2_score(y, y_pred))
                        ics.append(intercept)
                        ats.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'temperature_2m']))
                        wts.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'lake_mix_layer_temperature']))

        plt.sca(axes[i_y, 0])
        plt.scatter(xs, ys, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(ys, xs, frac=.125)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)
        plt.xlim([25, 55])
        plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
        plt.ylim([0, 1.5])
        plt.yticks([0, .5, 1, 1.5])
        if i_y == len(depth_ls) - 1:
            plt.xlabel('Absolute latitude', fontsize=9)
        if i_y == 2:
            plt.ylabel('Nutrient-Chl$a$ slope in shallow lakes\n(µg Chl$a$ / µg P)', fontsize=9)
        axes[i_y, 0].text(0.05, 0.95, "Average depth < {}m".format(depth),
                          transform=axes[i_y, 0].transAxes, fontsize=9, va='top', ha='left')

        used_data_df = filter_lakes(t_depth=depth)
        used_data_df['Unique Lake'] = used_data_df.index
        lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
        lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
        used_lake_ids = used_data_df.index.intersection(lake_climate_df.index)
        used_lake_cli_df = lake_climate_df.loc[used_lake_ids]
        merge_df = pd.concat([used_data_df, used_lake_cli_df], axis=1)

        y = merge_df['Chla (µg/L)'].values
        combo = ['ln', 'temperature_2m', 'mix_ln_at']
        merge_df['Latitude'] /= 60
        merge_df['ln'] /= 100
        merge_df['TN (µg/L)'] /= 3000
        merge_df['TP (µg/L)'] /= 100
        merge_df['temperature_2m'] /= 40
        merge_df['surface_solar_radiation_downwards_sum'] /= 800
        merge_df['mix_ln_at'] = merge_df['ln'] * merge_df['temperature_2m']
        xtick_dic = {
            'surface_solar_radiation_downwards_sum': 'Rad',
            'Latitude': 'Lat',
            'temperature_2m': 'Temp',
            'TN (µg/L)': 'TN',
            'TP (µg/L)': 'TP',
            'ln': 'LN',
            'mix_ln_at': 'Temp×LN',
        }
        X = merge_df[list(combo)].values
        n_features = X.shape[1]
        with pm.Model() as model:
            beta0 = pm.Normal("beta0", mu=0, sigma=10)
            betas = pm.Normal('betas', mu=0, sigma=10, shape=n_features)
            mu = beta0 + pm.math.dot(X, betas)
            sigma = pm.Exponential("sigma", 10)
            y_obs = pm.LogNormal("y_obs", mu=pm.math.log(mu), sigma=sigma, observed=y)
            trace = pm.sample(2000, tune=2000, return_inferencedata=True, target_accept=0.9, random_seed=119, chains=4)
            pm.compute_log_likelihood(trace)
            ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
            y_pred_samples = ppc.posterior_predictive["y_obs"].values

        y_pred = np.mean(y_pred_samples, axis=(0, 1))
        y_pred_lower = np.percentile(y_pred_samples, 5, axis=(0, 1))
        y_pred_upper = np.percentile(y_pred_samples, 95, axis=(0, 1))
        r2 = r2_score(y, y_pred)
        form = 'Chla ~ {}'.format(xtick_dic[combo[0]])
        if len(combo) > 1:
            for c in combo[1:]:
                form += ' + {}'.format(xtick_dic[c])

        plt.sca(axes[i_y, 1])
        plt.plot([y.min(), y.max()], [y.min(), y.max()], '-', lw=0.5, alpha=1, color='k', label="1:1 line")
        plt.scatter(y, y_pred, c=scat_c, edgecolors='k', s=30, linewidths=.5, alpha=.6, label='Posterior mean')
        plt.errorbar(y, y_pred, yerr=[y_pred - y_pred_lower, y_pred_upper - y_pred], fmt='', linestyle='', elinewidth=1, capsize=0, alpha=0.3, ecolor=scat_c, label="95% predictive interval")
        if i_y == len(depth_ls) - 1:
            plt.xlabel("Measured long-term growing-season Chla (µg/L)")
        if i_y == 2:
            plt.ylabel("Predicted long-term growing-season Chla (µg/L)")
        plt.xlim(0, 80)
        plt.ylim(0, 100)
        if i_y == 0:
            plt.legend(loc='lower right')
        axes[i_y, 1].text(0.05, 0.95, 'Bayesian model R$^2$: {:.2f} (n={})'.format(r2, len(y)), fontsize=9, ha='left',
                 va='top', color='k', transform=axes[i_y, 1].transAxes)
    plt.tight_layout()
    plt.savefig('Figs/FS sensitivity of depth filter.png', dpi=300)


def fs_different_bootstrap_paras():
    N = 250

    p_dic = {
        'Window width': {'depth': [3, 4, 5, 6, 7], 'lat': [10, 12.5, 15, 17.5, 20]},
        'Step size': {'depth': [0.05, 0.15, 0.25, 0.35, 0.5], 'lat': [0.1, 0.3, 0.5, 0.7, 1]},
        'Sample number': {'depth': [50, 150, 250, 350, 500], 'lat': [50, 150, 250, 350, 500]},
    }

    unit_dic = {
        'Window widthdepth': 'm',
        'Window widthlat': '°',
        'Step sizedepth': 'm',
        'Step sizelat': '°',
        'Sample numberdepth': '',
        'Sample numberlat': '',
    }

    for name, dic in p_dic.items():

        fig, axes = plt.subplots(5, 2, figsize=(8, 10), sharex='col')

        for i_y, v in enumerate(dic['depth']):
            mean_data_df = filter_lakes(n_year=5, t_depth=25, if_mean=True, if_filter=False)

            lat_mini, length, step, n, n_sam = -4, 5, .25, 80, N
            distance = step * n
            if name == 'Window width':
                length = v
            if name == 'Step size':
                step = v
                n = int(distance / step)
            if name == 'Sample number':
                n_sam = v

            lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
            xs, ys, rs, fs, ics, gis, gls, gips, ns, ats, wts = [], [], [], [], [], [], [], [], [], [], []
            for lat_min, lat_max in lat_windows:
                for i in range(n_sam):
                    samples = sample_lakes_in_depth_band(mean_data_df, lat_min, lat_max, n=30, ln_bins=10, random_state=911 + i)
                    if len(samples) >= 10:
                        x, y = samples['ln'], samples['Chla (µg/L)']
                        x_with_const = sm.add_constant(x)
                        glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                        y_pred = glm_model.predict(x_with_const)
                        slope, p_value = glm_model.params.loc['ln'], glm_model.pvalues.loc['ln']
                        if p_value < 0.05:
                            fs.append(r2_score(y, y_pred))
                            ys.append(slope)
                            xs.append(samples['depth'].mean())

            plt.sca(axes[i_y, 0])
            xs = np.array(xs)
            ys = np.array(ys)
            plt.scatter(xs[xs < 6], ys[xs < 6], alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
            plt.scatter(xs[xs >= 6], ys[xs >= 6], alpha=.25, c=depth_color, edgecolors=depth_color, s=30, linewidths=.5)
            lowess_result = sm.nonparametric.lowess(ys, xs, frac=.5)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)

            plt.xlim([0, 20])
            plt.xticks([0, 5, 10, 15, 20])
            plt.ylim([.2, .8])
            plt.yticks([.2, .4, .6, .8])
            if i_y == 4:
                plt.xlabel('Lake average depth (m)', fontsize=9)
            if i_y == 2:
                plt.ylabel('Nutrient-Chl$a$ slope (µg Chl$a$ / µg P)', fontsize=9)
            axes[i_y, 0].text(0.05, 0.95, "{} = {}{}".format(name, v, unit_dic[name+'depth']),
                              transform=axes[i_y, 0].transAxes, fontsize=9, va='top', ha='left')


        for i_y, v in enumerate(dic['lat']):
            used_data_df = filter_lakes(n_year=5, t_depth=6)
            used_data_df['Unique Lake'] = used_data_df.index
            valid_lakes = used_data_df.index
            lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
            lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
            cli_lake_ids = [valid_lake for valid_lake in valid_lakes if valid_lake in lake_climate_df.index]
            used_lake_cli_df = lake_climate_df.loc[cli_lake_ids]

            lat_mini, length, step, n, n_sam = 15, 15, .5, 70, N
            distance = step * n
            if name == 'Window width':
                length = v
            if name == 'Step size':
                step = v
                n = int(distance / step)
            if name == 'Sample number':
                n_sam = v

            lat_windows = [(lat_mini + i * step, lat_mini + i * step + length) for i in range(n)]
            xs, ys, fs, ics, ats, wts = [], [], [], [], [], []
            for lat_min, lat_max in lat_windows:
                for i in range(n_sam):
                    samples = sample_lakes_in_lat_band(used_data_df, lat_min, lat_max, n=30, tp_bins=10, random_state=911 + i + int(lat_min))
                    if len(samples) >= 10:
                        x, y = samples['ln'], samples['Chla (µg/L)']
                        x_with_const = sm.add_constant(x)
                        glm_model = sm.GLM(y, x_with_const, family=Gamma(link=identity())).fit()
                        y_pred = glm_model.predict(x_with_const)
                        slope, intercept, p_value = glm_model.params.loc['ln'], glm_model.params.loc['const'], \
                            glm_model.pvalues.loc['ln']
                        if p_value < 0.05:
                            xs.append(samples['Latitude'].mean())
                            ys.append(slope)
                            fs.append(r2_score(y, y_pred))
                            ics.append(intercept)
                            ats.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'temperature_2m']))
                            wts.append(np.mean(used_lake_cli_df.loc[samples['Unique Lake'], 'lake_mix_layer_temperature']))

            plt.sca(axes[i_y, 1])
            plt.scatter(xs, ys, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
            lowess_result = sm.nonparametric.lowess(ys, xs, frac=.125)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)
            plt.xlim([25, 55])
            plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
            plt.ylim([0, 1.5])
            plt.yticks([0, .5, 1, 1.5])
            if i_y == 4:
                plt.xlabel('Absolute latitude', fontsize=9)
            if i_y == 2:
                plt.ylabel('Nutrient-Chl$a$ slope in shallow lakes\n(µg Chl$a$ / µg P)', fontsize=9)
            axes[i_y, 1].text(0.05, 0.95, "{} = {}{}".format(name, v, unit_dic[name+'lat']),
                              transform=axes[i_y, 1].transAxes, fontsize=9, va='top', ha='left')

        plt.tight_layout()
        plt.savefig('Figs/FS sensitivity of bootstrap {}.png'.format(name), dpi=300)


if __name__ == '__main__':
    pass