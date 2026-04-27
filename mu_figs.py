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


def fig1():
    N = 250

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 4], width_ratios=[4, 4], wspace=0.3, hspace=0.3, left=0.08, right=0.92, bottom=0.08, top=0.95)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_axes([0.2, 0.08, 0.7, 0.42])

    mean_data_df = filter_lakes(n_year=5, t_depth=25, if_mean=True, if_filter=False)
    print('used to fit model all', len(mean_data_df))

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
        print('sampled', lat_min, lat_max)

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


    # Shallow lakes
    used_data_df = filter_lakes(n_year=5, t_depth=6)
    used_data_df['Unique Lake'] = used_data_df.index

    lat_mini, length, step, n, n_sam = 15, 15, .5, 70, N
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
                slope, intercept, p_value = glm_model.params.loc['ln'], glm_model.params.loc['const'], glm_model.pvalues.loc['ln']
                if p_value < 0.05:
                    xs.append(samples['Latitude'].mean())
                    ys.append(slope)
                    fs.append(r2_score(y, y_pred))
                    ics.append(intercept)
        print('sampled', lat_min)

    plt.sca(ax2)
    plt.scatter(xs, ys, alpha=.25, c=scat_c, edgecolors=scat_c, s=30, linewidths=.5)
    lowess_result = sm.nonparametric.lowess(ys, xs, frac=.125)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], line_c, linewidth=2, zorder=100)
    plt.xlim([25, 55])
    plt.xticks([25, 35, 45, 55], ['25°', '35°', '45°', '55°'])
    plt.ylim([0, 1.5])
    plt.yticks([0, .5, 1, 1.5])
    plt.xlabel('Absolute latitude', fontsize=9)
    plt.ylabel('Nutrient-Chl$a$ slope in shallow lakes\n(µg Chl$a$ / µg P)', fontsize=9)
    ax2.text(0.95, 0.95, "b", transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')


    plt.sca(ax3)
    used_data_df = filter_lakes(n_year=5, t_depth=6)
    used_data_df = calc_ln(used_data_df)
    used_data_df['Unique Lake'] = used_data_df.index
    lake_climate_df = pd.read_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_climate.csv'), header=0, index_col=0)
    lake_climate_df = fill_nan_climate(lake_climate_df, used_data_df)
    used_lake_ids = used_data_df.index.intersection(lake_climate_df.index)
    used_lake_cli_df = lake_climate_df.loc[used_lake_ids]
    merge_df = pd.concat([used_data_df, used_lake_cli_df], axis=1)
    merge_df['mix'] = merge_df['ln'] * merge_df['temperature_2m']

    X_names = ['ln', 'temperature_2m', 'mix']
    X = merge_df[X_names].values
    y = merge_df['Chla (µg/L)'].values

    with pm.Model() as model_lognormal:
        beta0 = pm.Normal("beta0", mu=1, sigma=10)
        betas = pm.Normal('betas', mu=1, sigma=10, shape=len(X_names))
        sigma = pm.Exponential("sigma", 10)
        mu = beta0 + pm.math.dot(X, betas)
        y_obs = pm.LogNormal("y_obs", mu=pm.math.log(mu), sigma=sigma, observed=y)
        trace = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=119, chains=4)
        pm.compute_log_likelihood(trace)

    beta0_samples = trace.posterior["beta0"].values.flatten()
    betas_samples = trace.posterior["betas"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(0.001, 40, 100, dtype=np.float32),
        np.linspace(0, 200, 100, dtype=np.float32),
    )
    grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel(), (x1_grid * x2_grid).ravel()]
    N_new = grid_points.shape[0]
    n_samples = len(beta0_samples)
    x1_new = np.array(grid_points[:, 0]).reshape(1, N_new)
    x2_new = np.array(grid_points[:, 1]).reshape(1, N_new)
    beta0_samples = beta0_samples.reshape(n_samples, 1).astype(np.float32)
    betas_samples = betas_samples.reshape(n_samples, 3).astype(np.float32)
    sigma_samples = sigma_samples.reshape(n_samples, 1).astype(np.float32)
    mu = beta0_samples + betas_samples[:, 0] * x1_new + betas_samples[:, 1] * x2_new + betas_samples[:, 2] * x1_new * x2_new
    y_pred_new = np.random.lognormal(mean=np.log(mu), sigma=sigma_samples)
    y_pred_mean = np.mean(y_pred_new, axis=0)

    y_pred_grid = y_pred_mean.reshape(x1_grid.shape)
    contour = plt.contourf(x1_grid, x2_grid, y_pred_grid, cmap='viridis', levels=50)
    cbar = plt.colorbar(contour)
    cbar.outline.set_linewidth(.5)
    cbar.set_label('Chl$a$ in shallow lakes (µg/L)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.tick_params(length=2, width=.5)
    plt.contour(x1_grid, x2_grid, y_pred_grid, levels=[12], colors=line_c, linewidths=2)
    ax3.text(0.5, 0.2, 'Chl$a$ = 12 µg/L', transform=ax3.transAxes, ha='left', va='top', color=line_c, fontsize=9)
    plt.ylim([0, 200])
    plt.yticks([0, 50, 100, 150, 200])
    plt.xlim([0, 40])
    plt.xticks([0, 10, 20, 30, 40])
    plt.xlabel('Growing-season temperature (°C)', fontsize=9)
    plt.ylabel('Nutrient (µg P/L)', fontsize=9)

    T_crit = 11.658291457286433
    ax3.axvline(T_crit, color='red', linestyle='-', linewidth=2)
    ax3.text(T_crit + 1, 0.9, 'Temperature = {:.1f} °C'.format(T_crit),
             transform=ax3.get_xaxis_transform(), ha='left', va='top', color='red', fontsize=9)
    ax3.text(0.95, 0.95, "c", transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig('Figs/F1.png', dpi=300)


def fig2():
    data_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col='Hylak_id')
    chla_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_mean_bayes.csv', index_col='Hylak_id')
    temp_df = load_hydrolakes_cmip6_temp()
    temp_df.columns = [str(col) + '_temp' for col in temp_df.columns]
    data_df = pd.concat([data_df, chla_df, temp_df], axis=1)
    print('All lake num', len(data_df))
    print('Shallow lake number ratio', len(data_df[data_df['depth'] < 6]), len(data_df[data_df['depth'] < 6]) / len(data_df))
    print('Shallow lake area ratio', data_df[data_df['depth'] < 6]['Lake_area'].sum() / data_df['Lake_area'].sum())
    data_df = data_df[data_df['depth'] < 6]

    print('Shallow lake num', len(data_df))
    filter_data_df = data_df.dropna(subset=['ln', '2020'])
    print('Lake ln&chl num', len(filter_data_df))
    print('used lake area', filter_data_df['Lake_area'].sum() / data_df[data_df['depth'] < 6]['Lake_area'].sum())

    filter_data_df['delta_chla'] = filter_data_df['2100'] - filter_data_df['2020']
    filter_data_df['delta_temp'] = filter_data_df['2100_temp'] - filter_data_df['2020_temp']

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[4, 3, 3], width_ratios=[8.5, 1.5], wspace=0.075, left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    gs_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], width_ratios=[7, 3], wspace=0.4)
    ax3 = fig.add_subplot(gs_bottom[0, 0])
    ax4 = fig.add_subplot(gs_bottom[0, 1])
    gs_mid = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], width_ratios=[7, 3], wspace=0.4)
    ax5 = fig.add_subplot(gs_mid[0, 0])
    ax6 = fig.add_subplot(gs_mid[0, 1])

    m = Basemap(projection='cyl', resolution='i', ax=ax1)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='lightgray', lake_color='lightgray', alpha=.5)
    m.drawmapboundary(fill_color='none', linewidth=0.5)

    parallels = np.arange(-80, 81, 20)
    meridians = np.arange(-180, 181, 60)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25, dashes=[5, 0], fontsize=7, color='lightgray')
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25, dashes=[5, 0], fontsize=7, color='lightgray')
    ax1.set_xlabel('Longitude', fontsize=8, labelpad=15)
    ax1.set_ylabel('Latitude', fontsize=8, labelpad=24)

    x, y = m(data_df['Pour_long'].values, data_df['Pour_lat'].values)
    m.scatter(
        x, y,
        c='#B0C4DE',
        marker='s',
        s=.1,
        edgecolor='none',
        alpha=1,
    )

    colors = [str(y) for y in range(1980, 2101)]
    color_dic = create_color_dict(colors, 'OrRd')
    color_dic['2020'] = '#FED9A6'

    bar_ticks = [0, 5, 10, 15, 20]
    bar_ticklabels = [str(t) for t in bar_ticks]
    bar_ticklabels[-1] = '>' + bar_ticklabels[-1]

    values = filter_data_df['delta_chla']
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
    ax1.set_aspect('auto')

    cax = ax1.inset_axes([0.6, 0.2, 0.2, 0.03])
    norm = mpl.colors.Normalize(vmin=bar_ticks[0], vmax=bar_ticks[-1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal', ticks=[2020, 2060, 2100])
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks(bar_ticks)
    cb.set_ticklabels(bar_ticklabels, fontsize=7)
    cb.set_label("ΔChl$a$ (µg/L)", fontsize=7)

    text_loc_dic = {'delta_chla': [0.1, 0.95, 'ΔChl$a$'], '2020': [0.15, 0.65, 'Chl$a$ (2020)'], '2100': [0.3, 0.35, 'Chl$a$ (2100)']}
    ax_inner = ax1.inset_axes([0.05, 0.2, 0.2, 0.4], zorder=100)
    for c_n, c in {'2020': color_dic['2020'], '2100': color_dic['2100'], 'delta_chla': '#287D8E'}.items():
        values = filter_data_df[c_n].values
        sns.kdeplot(values, fill=False, color=c, alpha=1, linewidth=1.5, ax=ax_inner)
        ax_inner.set_xlabel("µg/L", fontsize=7)
        ax_inner.set_ylabel('Density', fontsize=7)
        ax_inner.set_yticks([])
        ax_inner.set_xticks([0, 10, 20, 30, 40])
        ax_inner.set_xlim([0, 40])
        ax_inner.tick_params(axis='both', which='major', width=.5, length=1, labelsize=7)
        ax_inner.grid(False)
        ax_inner.spines['top'].set_visible(False)
        ax_inner.spines['right'].set_visible(False)
        ax_inner.set_facecolor('none')
        ax_inner.patch.set_alpha(0.0)
        ax_inner.text(*text_loc_dic[c_n], transform=ax_inner.transAxes, fontsize=7, va='top', ha='left', color=c)

    chla_ticks = [0, 10, 20, 30]
    years = [2020, 2100]
    width = 1
    lat_bins = np.arange(-90, 91, width)
    filter_data_df['lat_bin'] = pd.cut(filter_data_df['Pour_lat'], bins=lat_bins, include_lowest=True)
    for i_r, r in enumerate(years):
        grouped = filter_data_df.groupby('lat_bin', as_index=False).agg({
            '{}'.format(r): 'mean',
            'Pour_lat': 'mean'
        })
        grouped = grouped.dropna(axis=0)
        x, y = grouped['Pour_lat'].values, grouped['{}'.format(r)].values
        y = np.convolve(y, np.ones(5) / 5, mode='same')
        ax2.plot(y[:-2], x[:-2], lw=2, c=color_dic[str(r)], alpha=1, label=str(r))
    ax2.set_xlabel('Chl$a$ (µg/L)', fontsize=8)
    ax2.set_ylim([-90, 90])
    ax2.set_yticks(range(-80, 81, 20), [])
    ax2.set_xlim(chla_ticks[0], chla_ticks[-1])
    ax2.set_xticks(chla_ticks)
    ax2.tick_params(axis='both', labelsize=7, length=0)
    ax2.legend(loc='lower left', fontsize=7)

    ax2_twin = ax2.twiny()
    delta_chla_ticks = [1, 3, 5, 7]
    grouped = filter_data_df.groupby('lat_bin', as_index=False).agg({
        'delta_chla': 'mean',
        'Pour_lat': 'mean'
    })
    grouped = grouped.dropna(axis=0)
    x, y = grouped['Pour_lat'].values, grouped['delta_chla'].values
    y = np.convolve(y, np.ones(5) / 5, mode='same')
    ax2_twin.plot(y[:-2], x[:-2], color='#287D8E', linewidth=2)
    ax2_twin.set_xlabel('ΔChl$a$ (µg/L)', fontsize=8)
    ax2_twin.tick_params(axis='x', labelsize=7, top=True, bottom=False, length=0)
    ax2_twin.xaxis.set_label_position('top')
    ax2_twin.set_xlim(delta_chla_ticks[0], delta_chla_ticks[-1])
    ax2_twin.set_xticks(delta_chla_ticks)

    width = 2
    lat_bins = np.arange(-90, 91, width)
    lat_labels = lat_bins[:-1]
    all_counts, _ = np.histogram(data_df['Pour_lat'], bins=lat_bins)
    ax3_tx = ax3.twinx()
    ax3.bar(lat_labels, all_counts, color='#B0C4DE', edgecolor='none', alpha=0.8, width=width * .75)
    bottom = None
    idx = None
    for col in [str(y) for y in range(2020, 2101)]:
        year_ss = filter_data_df[col]
        risk_ss = year_ss[year_ss > 12]
        if idx is None:
            risk_idx = risk_ss.index
        else:
            risk_idx = risk_ss.index.difference(idx)
        year_df = filter_data_df.loc[risk_idx]
        year_counts, _ = np.histogram(year_df['Pour_lat'], bins=lat_bins)
        if col == '2020':
            label = 'Existing bloom-vulnerable lakes in 2020'
        elif col == '2100':
            label = 'Newly emerged bloom-vulnerable lakes by 2100'
        else:
            label = None
        ax3.bar(lat_labels, year_counts, color=color_dic[col], bottom=bottom, edgecolor='none', alpha=1, width=width * .75, label=label)
        if bottom is None:
            bottom = year_counts * 0
        bottom += year_counts
        idx = risk_ss.index
        if col in ['2020', '2090']:
            ax3_tx.plot(lat_labels, bottom / all_counts, lw=2, c=color_dic[col], alpha=1, label=col if col == '2020' else '2100')
    plt.xticks(range(-80, 81, 20), ['80°S', '60°S', '40°S', '20°S', '0°', '20°N', '40°N', '60°N', '80°N'])
    plt.xlim(-80, 80)
    ax3.set_yticks([0, 3e4, 6e4, 9e4, 12e4], [0, 3, 6, 9, 12])
    ax3.set_ylim([0, 12e4])
    ax3.set_ylabel('Number of shallow lakes (×10$^4$)', fontsize=8)
    ax3.set_xlabel('Latitude', fontsize=8)
    ax3.grid(axis='both', alpha=0.4, color='grey', lw=0.4)
    ax3.tick_params(labelsize=7, width=.5, length=2)
    for spine in ax3.spines.values():
        spine.set_linewidth(.5)
    ax3_tx.set_ylim([0, 0.6])
    ax3_tx.set_yticks([0, 0.15, 0.3, 0.45, 0.6], [0, 15, 30, 45, 60])
    ax3_tx.set_ylabel('Proportion of bloom-vulnerable lakes (%)', fontsize=8)
    ax3_tx.tick_params(labelsize=7, width=.5, length=2)
    for spine in ax3_tx.spines.values():
        spine.set_linewidth(.5)
    ax3.legend(loc='upper left', fontsize=6)
    ax3_tx.legend(loc='upper right', fontsize=7)

    pred_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_mean_bayes.csv', index_col='Hylak_id')
    low_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_low_bayes.csv', index_col='Hylak_id')
    high_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_high_bayes.csv', index_col='Hylak_id')

    risks = []
    lows, highs, lats = [], [], []
    years = [y for y in range(2020, 2101)]
    base_idx = pred_df[pred_df['2020'] > 12].index
    for year in years:
        year_ss = pred_df[str(year)]
        low_ss = low_df[str(year)]
        high_ss = high_df[str(year)]
        n_risk = len(year_ss[year_ss > 12])
        n_low = len(low_ss[low_ss > 12])
        n_high = len(high_ss[high_ss > 12])
        risks.append(n_risk)
        lows.append(n_low)
        highs.append(n_high)
        new_idx = year_ss[year_ss > 12].index.difference(base_idx)
        new_idx = new_idx.intersection(filter_data_df.index)
        lat_ss = filter_data_df.loc[new_idx]['Pour_lat']
        lats.append(lat_ss[lat_ss > 0])
        print('risk lake ratio at year {}: {} {} {}'.format(year, n_risk / len(data_df), n_low / len(data_df), n_high / len(data_df)))

    ax4.fill_between(
        years,
        lows,
        highs,
        color=color_dic['2100'],
        alpha=0.2,
        label='95% Cred-Int'
    )
    ax4.plot(years, risks, lw=2, c=color_dic['2100'], alpha=1)
    ax4.set_yticks([10e4, 15e4, 20e4, 25e4, 30e4], [10, 15, 20, 25, 30])
    ax4.set_ylim([10e4, 30e4])
    ax4.set_xticks(range(2020, 2101, 20))
    ax4.set_xlim([2020, 2100])
    ax4.set_ylabel('Number of bloom-vulnerable lakes (×10$^4$)', fontsize=8)
    ax4.set_xlabel('Year', fontsize=8)
    ax4.grid(axis='both', alpha=0.4, color='grey', lw=0.4)
    ax4.tick_params(labelsize=7, width=.5, length=2)
    for spine in ax4.spines.values():
        spine.set_linewidth(.5)
    ax4.legend(loc='upper left', fontsize=7)

    width = 1
    temp_bins = np.arange(0, 41, width)
    temp_labels = temp_bins[:-1]
    for y in [2020, 2100]:
        all_counts, _ = np.histogram(data_df['{}_temp'.format(y)], bins=temp_bins)
        ax5.bar(temp_labels, all_counts, color=color_dic[str(y)], edgecolor='none', alpha=1 if y == 2020 else 0.25, width=width * .75 if y == 2020 else width * .75, label=str(y))
    ax5.set_xticks(range(0, 41, 5))
    ax5.set_xlim(0, 40)
    ax5.set_yticks([0, 3e4, 6e4, 9e4, 12e4], [0, 3, 6, 9, 12])
    ax5.set_ylim([0, 12e4])
    ax5.set_xlabel('Growing-season temperature (°C)', fontsize=8)
    ax5.set_ylabel('Number of shallow lakes (×10$^4$)', fontsize=8)
    ax5.grid(axis='both', alpha=0.4, color='grey', lw=0.4)
    ax5.tick_params(labelsize=7, width=.5, length=2)
    ax5.legend(loc='upper left', fontsize=7)

    ax5_tx = ax5.twinx()
    min_t = 11.658291457286433
    for y in [2020, 2100]:
        df_sorted = data_df.sort_values('{}_temp'.format(y)).reset_index(drop=True)
        df_sorted['cum_prop'] = np.arange(1, len(df_sorted) + 1) / len(df_sorted)
        ax5_tx.plot(df_sorted['{}_temp'.format(y)], df_sorted['cum_prop'], lw=2, c=color_dic[str(y)], alpha=1, label=str(y))
    ax5_tx.set_xticks(range(0, 41, 5))
    ax5_tx.set_xlim(0, 40)
    ax5_tx.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax5_tx.set_ylim([0, 1])
    ax5_tx.set_ylabel('Cumulative proportion (%)', fontsize=8)
    ax5_tx.tick_params(labelsize=7, width=.5, length=2)
    ax5_tx.legend(loc='upper right', fontsize=7)
    T_crit = min_t
    ax5.axvline(T_crit, color='red', linestyle='-', linewidth=2)
    ax5_tx.text(T_crit + 0.5, 0.03, '{:.1f} °C'.format(T_crit), transform=ax5_tx.get_xaxis_transform(), ha='left', va='bottom', color='red', fontsize=8)

    risks = []
    lows, highs = [], []
    years = [y for y in range(2020, 2101)]
    for year in years:
        n_risk = len(data_df[data_df['{}_temp'.format(year)] > min_t])
        risks.append(n_risk)
    ax6.plot(years, risks, lw=2, c=color_dic['2100'], alpha=1)
    ax6.fill_between(
        years,
        lows,
        highs,
        color=color_dic['2100'],
        alpha=0.2,
        label='95% Cred-Int',
    )
    ax6.set_xticks(range(2020, 2101, 20))
    ax6.set_xlim([2020, 2100])
    ax6.set_yticks([4e5, 6e5, 8e5, 10e5, 12e5], [4, 6, 8, 10, 12])
    ax6.set_ylim([4e5, 12e5])
    ax6.set_ylabel('Number of nutrient-sensitive lakes (×10$^5$)', fontsize=8)
    ax6.set_xlabel('Year', fontsize=8)
    ax6.grid(axis='both', alpha=0.4, color='grey', lw=0.4)
    ax6.tick_params(labelsize=7, width=.5, length=2)
    ax6.legend(loc='upper left', fontsize=7)

    ax1.text(0.02, 0.95, "a", transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    ax2.text(0.05, 0.95, "b", transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    ax5.text(0.03, 0.8, "c", transform=ax5.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    ax6.text(0.05, 0.85, "d", transform=ax6.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    ax3.text(0.03, 0.8, "e", transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    ax4.text(0.05, 0.85, "f", transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    plt.savefig('Figs/F2.png', dpi=300)


def fig3():
    data_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col='Hylak_id')
    chla_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_future_chl_mean_bayes.csv', index_col='Hylak_id')
    temp_df = load_hydrolakes_cmip6_temp()
    temp_df.columns = [str(col) + '_temp' for col in temp_df.columns]
    data_df = pd.concat([data_df, chla_df, temp_df], axis=1)
    print('All lake num', len(data_df))
    print('Shallow lake number ratio', len(data_df[data_df['depth'] < 6]), len(data_df[data_df['depth'] < 6]) / len(data_df))
    print('Shallow lake area ratio', data_df[data_df['depth'] < 6]['Lake_area'].sum() / data_df['Lake_area'].sum())
    data_df = data_df[data_df['depth'] < 6]

    plt.figure(figsize=(8, 4))
    ax3_tx = plt.gca()
    ax3 = ax3_tx.twinx()

    colors = [str(y) for y in range(1980, 2101)]
    color_dic = create_color_dict(colors, 'OrRd')
    color_dic['2020'] = '#FED9A6'
    color_dic['2060'] = color_dic['2070']
    line_label_dic = {
        '2020_NT': '2020',
        '2040_NT': '2100 (SSP1-2.6)',
        '2060_NT': '2100 (SSP2-4.5)',
        '2100_NT': '2100 (SSP5-8.5)',
    }
    width = 2
    lat_bins = np.arange(-90, 91, width)
    data_df['lat_bin'] = pd.cut(data_df['Pour_lat'], bins=lat_bins, include_lowest=True)
    for i_r, r in enumerate(['2020_NT', '2040_NT', '2060_NT', '2100_NT']):
        grouped = data_df.groupby('lat_bin', as_index=False).agg({
            r: 'mean',
            'Pour_lat': 'mean'
        })
        ax3.plot(grouped['Pour_lat'], grouped[r], lw=1.5, c=color_dic[r[:4]], alpha=1, label=line_label_dic[r], zorder=1000)
    ax3.set_ylim(0, 200)
    ax3.set_yticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax3.set_ylabel('Nutrient threshold (µg P/L)', fontsize=9)
    width = 2
    lat_bins = np.arange(-90, 91, width)
    lat_labels = lat_bins[:-1]
    all_counts, _ = np.histogram(data_df['Pour_lat'], bins=lat_bins)
    ax3_tx.bar(lat_labels, all_counts, color='#B0C4DE', edgecolor='none', alpha=0.6, width=width * .75, label='Shallow lakes', zorder=0)
    ax3_tx.set_yticks([0, 2e4, 4e4, 6e4, 8e4, 10e4, 12e4, 14e4, 16e4], [0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax3_tx.set_ylim([0, 16e4])
    ax3_tx.set_ylabel('Number of shallow lakes (×10$^4$)', fontsize=9)
    ax3.legend(loc='upper right', fontsize=8, bbox_to_anchor=(0.9, 0.98))
    ax3_tx.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0.02, 0.98))

    ax3.set_xlabel('Latitude', fontsize=9)
    ax3_tx.set_xlabel('Latitude', fontsize=9)
    plt.xticks(range(-80, 81, 20), ['80°S', '60°S', '40°S', '20°S', '0°', '20°N', '40°N', '60°N', '80°N'])
    plt.xlim(-80, 80)

    plt.tight_layout()
    plt.savefig('Figs/F3.png', dpi=300)


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()