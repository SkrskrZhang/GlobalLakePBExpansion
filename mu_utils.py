import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.tri as tri
import chardet
import matplotlib.colors as mcolors

from mu_config import *


def get_lake_meta():
    lake_mor_df = pd.read_csv('open_source_dataset/AWQDFGL-main/Lake_Morphology.csv', header=0, encoding='latin1')
    lake_mor_df = lake_mor_df.set_index(lake_mor_df['Unique Lake'])
    lake_alt_df = pd.read_csv('open_source_dataset/AWQDFGL-main/Lake_Altitude.csv', header=0, encoding='latin1').loc[:, ['Unique Lake', 'Latitude', 'Longitude', 'Altitude (m)']]
    lake_alt_df = lake_alt_df.groupby('Unique Lake').mean()

    china_lake_meta_df = pd.read_csv('open_source_dataset\AWQDFGL-main\china_lake_meta.csv', header=0)
    china_lake_meta_df['Unique Lake'] = 'CHN' + china_lake_meta_df['Unique Lake'].astype('str')
    china_lake_meta_df = china_lake_meta_df.set_index(china_lake_meta_df['Unique Lake'])

    lake_meta_df = pd.concat([lake_mor_df, lake_alt_df], axis=1)
    lake_meta_df = pd.concat([lake_meta_df, china_lake_meta_df], axis=0)

    lake_meta_df.to_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_meta.csv'), index=False)
    return lake_meta_df


def get_lake_wq(if_china=True, if_use=True):
    awq_data_df = pd.read_csv('open_source_dataset/AWQDFGL-main/Lake_Metadata.csv', header=0, encoding='latin1', dtype={7: "string", 9: "string", 11: "string",
                                                                                                                    15: "string", 17: "string", 22: "string", 24: "string"})
    awq_data_df['sampledate'] = pd.to_datetime(awq_data_df['sampledate'])
    if if_china:
        chn_data_df = pd.read_csv('open_source_dataset\AWQDFGL-main\china_lake_wq.csv', header=0)
        chn_data_df['sampledate'] = pd.to_datetime(chn_data_df['sampledate'])

        china_lake_meta_df = pd.read_csv('open_source_dataset\AWQDFGL-main\china_lake_meta.csv', header=0)
        china_lake_meta_df['Unique Lake'] = 'CHN' + china_lake_meta_df['Unique Lake'].astype('str')
        chn_data_df = chn_data_df[chn_data_df['Unique Lake'].isin(china_lake_meta_df['Unique Lake'])]
        all_data_df = pd.concat([awq_data_df, chn_data_df], axis=0)
        all_data_df.to_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_lake_wq.csv'), index=False)
    else:
        all_data_df = awq_data_df
    return all_data_df


def calc_std_tn(tp):
    return np.power(10, 0.8073 * np.log10(tp) + 1.5015)


def calc_ln(df):
    tp, tn = df['TP (µg/L)'], df['TN (µg/L)']
    std_tn = calc_std_tn(tp)
    std_n2p = std_tn / tp
    # df['Limit_Type'] = np.where(tn/tp < std_n2p, 'N', 'P')
    df['ln'] = np.where(tn/tp > std_n2p, tp, tn / std_n2p)
    df['ln_t'] = np.where(tn/tp > std_n2p, 0, 1)
    return df


def has_consecutive_years(years, min_len):
    longest = cur_len = 1
    for i in range(1, len(years)):
        if years[i] == years[i - 1] + 1:
            cur_len += 1
            longest = max(longest, cur_len)
        else:
            cur_len = 1
    return longest >= min_len


def has_years(years, min_len):
    return len(years) >= min_len


def has_constant_years(years, c_len):
    return len(years) == c_len


def has_bound_years(years, c_len, buffer=1):
    return max(2, c_len - buffer) <= len(years) <= c_len + buffer


def filter_summer_data(data_df, c_date='sampledate', c_lat='Latitude', t_lat=5):
    data_df[c_date] = pd.to_datetime(data_df[c_date])
    data_df['year'] = data_df[c_date].dt.year
    data_df['month'] = data_df[c_date].dt.month
    data_df['Abs_lat'] = np.abs(data_df[c_lat])
    o_data_df = data_df[data_df['Abs_lat'] <= t_lat]
    n_data_df = data_df[data_df[c_lat] > t_lat]
    n_summer_data_df = n_data_df[n_data_df['month'].isin(summer_months)]
    s_data_df = data_df[data_df[c_lat] < -t_lat]
    s_summer_data_df = s_data_df[s_data_df['month'].isin(summer_months_s)]
    data_df = pd.concat([o_data_df, n_summer_data_df, s_summer_data_df], axis=0)
    return data_df


def sample_lakes_in_lat_band(df, lat_min, lat_max, n, tp_bins, random_state=None):
    subset = df[(df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max)]
    if subset.empty:
        return pd.DataFrame()
    subset = subset.sort_values('ln')
    subset['tp_bin'] = pd.qcut(subset['ln'], q=tp_bins, duplicates='drop')
    sampled = []
    if len(subset) < 5:
        return sampled
    else:
        counts = np.repeat(n // tp_bins, tp_bins)
        counts[: n % tp_bins] += 1
        for bin_idx, (bin_label, group) in enumerate(subset.groupby('tp_bin')):
            if len(group) > 0 and bin_idx != 0 and bin_idx != tp_bins-1:
                for i in range(counts[bin_idx]):
                    sample = group.sample(n=1, random_state=random_state)
                    sampled.append(sample)
                    random_state += 100
        return pd.concat(sampled, ignore_index=True) if len(sampled) > 0 else sampled


def sample_lakes_in_ln_band(df, ln_min, ln_max, n, ln_bins, random_state=None):
    subset = df[(df['ln'] >= ln_min) & (df['ln'] <= ln_max)]
    if subset.empty:
        return pd.DataFrame()
    subset = subset.sort_values('temperature_2m')
    subset['t_bin'] = pd.qcut(subset['temperature_2m'], q=ln_bins, duplicates='drop')
    sampled = []
    if len(subset) < 5:
        return sampled
    else:
        counts = np.repeat(n // ln_bins, ln_bins)
        counts[: n % ln_bins] += 1
        for bin_idx, (bin_label, group) in enumerate(subset.groupby('t_bin')):
            if len(group) > 0 and bin_idx != 0 and bin_idx != ln_bins-1:
                for i in range(counts[bin_idx]):
                    sample = group.sample(n=1, random_state=random_state)
                    sampled.append(sample)
                    random_state += 100
        return pd.concat(sampled, ignore_index=True) if len(sampled) > 0 else sampled


def sample_lakes_in_depth_band(df, ln_min, ln_max, n, ln_bins, random_state=None):
    subset = df[(df['depth'] >= ln_min) & (df['depth'] <= ln_max)]
    if subset.empty:
        return pd.DataFrame()
    subset = subset.sort_values('ln')
    subset['t_bin'] = pd.qcut(subset['ln'], q=ln_bins, duplicates='drop')
    sampled = []
    if len(subset) < 5:
        return sampled
    else:
        counts = np.repeat(n // ln_bins, ln_bins)
        counts[: n % ln_bins] += 1
        for bin_idx, (bin_label, group) in enumerate(subset.groupby('t_bin')):
            if len(group) > 0 and bin_idx != 0 and bin_idx != ln_bins-1:
                for i in range(counts[bin_idx]):
                    sample = group.sample(n=1, random_state=random_state)
                    sampled.append(sample)
                    random_state += 100 # no same samples
        return pd.concat(sampled, ignore_index=True) if len(sampled) > 0 else sampled


def fill_nan_climate(cli_df, lake_df):
    lake_ids = lake_df.index
    cli_lake_ids = [lake_id for lake_id in lake_ids if lake_id in cli_df.index]
    nan_lake_ids = [lake_id for lake_id in lake_ids if lake_id not in cli_df.index]
    print('no climate lakes', nan_lake_ids)
    append_df = pd.DataFrame(index=nan_lake_ids, columns=cli_df.columns, dtype=np.float64)
    if len(nan_lake_ids) > 0:
        for col in cli_df.columns:
            x = lake_df.loc[cli_lake_ids, 'Latitude'].values
            y = cli_df.loc[cli_lake_ids, col].values
            regr = RandomForestRegressor(n_estimators=100, random_state=911)
            regr.fit(x.reshape(-1, 1), y)
            y_pred = regr.predict(lake_df.loc[nan_lake_ids, 'Latitude'].values.reshape(-1, 1))
            append_df.loc[nan_lake_ids, col] = y_pred
        return pd.concat([cli_df, append_df], axis=0)
    else:
        return cli_df


def filter_lakes(n_year=5, t_depth=6, if_mean=True, if_filter=True):
    data_df = get_lake_wq(if_china=True)

    data_df = data_df.dropna(subset=['sampledate', 'Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'])
    print('filter has all data records', len(data_df))
    print('filter has all data lakes', len(data_df.groupby('Unique Lake').mean(numeric_only=True)))

    data_df = data_df[((data_df['Chla (µg/L)'] < 200) & (data_df['Chla (µg/L)'] > 1)) | (data_df['Chla (µg/L)'].isna())]
    data_df = data_df[((data_df['TP (µg/L)'] < 200) & (data_df['TP (µg/L)'] > 1)) | (data_df['TP (µg/L)'].isna())]
    print('filter small data records', len(data_df))
    print('filter small data lakes', len(data_df.groupby('Unique Lake').mean(numeric_only=True)))

    meta_df = get_lake_meta()

    for c in meta_df.columns:
        if c not in data_df.columns:
            data_df[c] = data_df['Unique Lake'].map(meta_df[c])
    data_df.to_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'all_data.csv'))

    summer_data_df = filter_summer_data(data_df).dropna(subset=['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'])
    summer1980_data_df = summer_data_df[summer_data_df['year'] >= 1980]
    print('n_summer_chl_tn_tp_records', len(summer1980_data_df))
    print('n_lake_chl_tn_tp_records', len(summer1980_data_df.groupby('Unique Lake').mean(numeric_only=True)))

    year_summer_data_df = summer1980_data_df.groupby(['Unique Lake', 'year']).mean(numeric_only=True)
    year_summer_data_df = year_summer_data_df.dropna(subset=['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'])
    lake_years = year_summer_data_df.groupby(level=0).apply(lambda x: sorted(x.index.get_level_values(1)))

    n_year = n_year
    valid_lakes = [lake for lake, years in lake_years.items() if has_years(years, n_year)]
    print('n year > {} lakes'.format(n_year), len(valid_lakes))

    # Filter lake depth
    depth_filter = t_depth
    year_summer_data_df = year_summer_data_df.reset_index()
    year_summer_greater5_data_df = year_summer_data_df[year_summer_data_df['Unique Lake'].isin(valid_lakes)]
    year_summer_greater5_mor_df = meta_df.loc[valid_lakes].dropna(subset=['MEAN DEPTH (m)', 'Max Depth (m)'], how='all')
    year_summer_greater5_mor_df['depth'] = year_summer_greater5_mor_df['MEAN DEPTH (m)']
    year_summer_greater5_mor_df['depth'] = year_summer_greater5_mor_df['depth'].fillna(year_summer_greater5_mor_df['Max Depth (m)'] * .5)
    valid_lakes = year_summer_greater5_mor_df[year_summer_greater5_mor_df['depth'] < depth_filter].index
    print('n depth < {}m lakes'.format(depth_filter), len(valid_lakes))
    year_summer_greater5_shallow6_data_df = year_summer_greater5_data_df[year_summer_greater5_data_df['Unique Lake'].isin(valid_lakes)]
    year_summer_greater5_shallow6_mean_data_df = year_summer_greater5_shallow6_data_df.groupby('Unique Lake').mean()
    year_summer_greater5_shallow6_mean_data_df = calc_ln(year_summer_greater5_shallow6_mean_data_df)
    year_summer_greater5_shallow6_mean_data_df.loc[:, 'depth'] = year_summer_greater5_mor_df.loc[year_summer_greater5_shallow6_mean_data_df.index, 'depth']
    year_summer_greater5_shallow6_mean_data_df.to_csv(os.path.join('open_source_dataset/AWQDFGL-main', 'filter_data.csv'))

    print('long-term records', len(summer1980_data_df[summer1980_data_df['Unique Lake'].isin(valid_lakes)]))
    print('long-term lakes', len(summer1980_data_df[summer1980_data_df['Unique Lake'].isin(valid_lakes)].groupby('Unique Lake').mean(numeric_only=True)))

    if if_mean:
        return year_summer_greater5_shallow6_mean_data_df
    else:
        year_summer_greater5_shallow6_data_df = calc_ln(year_summer_greater5_shallow6_data_df)
        year_summer_greater5_shallow6_data_df['depth'] = year_summer_greater5_shallow6_data_df['Unique Lake'].map(year_summer_greater5_shallow6_mean_data_df['depth'])
        return year_summer_greater5_shallow6_data_df


def filter_lakes_with_rse(mean_df, t_chl, t_dic=None):
    cv_df = pd.read_csv('Results/cv.csv', index_col=0, header=0)
    chl_idx = cv_df[cv_df['rse_{}'.format('Ch')] <= t_chl].index
    if t_dic is not None:
        for c, t in t_dic.items():
            c_idx = cv_df[cv_df['rse_{}'.format(c)] <= t].index
            chl_idx = chl_idx.intersection(c_idx)
    return mean_df.loc[mean_df.index.intersection(chl_idx)]


def create_color_dict(unique_values, colormap_name):
    colormap = plt.get_cmap(colormap_name)
    colors_indices = np.linspace(0, 1, len(unique_values))
    color_dict = {}
    for i, value in enumerate(unique_values):
        rgba_color = colormap(colors_indices[i])
        hex_color = mcolors.to_hex(rgba_color)
        color_dict[value] = hex_color
    return color_dict


def scale_data(data, min=None, max=None, eps=1e-6):
    min = data.min() if min is None else min
    max = data.max() if max is None else max
    return (data - min + eps) / (max - min + eps)


def load_hydrolakes_cmip6_temp():
    save_path = 'open_source_dataset/AWQDFGL-main/GEE_hylak_climate/hydrolakes_summer_temp_ssp585_2015-2100.csv'

    if not os.path.exists(save_path):
        hylak_df = pd.read_csv('open_source_dataset/AWQDFGL-main/hydro_lake_data.csv', header=0, index_col=0)
        n_index = hylak_df[hylak_df['Pour_lat'] >= 0].index
        s_index = hylak_df[hylak_df['Pour_lat'] < 0].index
        n_df1 = pd.read_csv('open_source_dataset/AWQDFGL-main/GEE_hylak_climate/hydrolakes_summer_temp_ssp585_2015-2050_N.csv', header=0, index_col=('year', 'Hylak_id'))[['mean']]
        n_df2 = pd.read_csv('open_source_dataset/AWQDFGL-main/GEE_hylak_climate/hydrolakes_summer_temp_ssp585_2051-2100_N.csv', header=0, index_col=('year', 'Hylak_id'))[['mean']]
        n_merge_df = pd.concat([n_df1, n_df2], axis=0)
        n_merge_df = n_merge_df[n_merge_df.index.get_level_values(1).isin(n_index)]
        s_df1 = pd.read_csv('open_source_dataset/AWQDFGL-main/GEE_hylak_climate/hydrolakes_summer_temp_ssp585_2015-2050_S.csv', header=0, index_col=('year', 'Hylak_id'))[['mean']]
        s_df2 = pd.read_csv('open_source_dataset/AWQDFGL-main/GEE_hylak_climate/hydrolakes_summer_temp_ssp585_2051-2100_S.csv', header=0, index_col=('year', 'Hylak_id'))[['mean']]
        s_merge_df = pd.concat([s_df1, s_df2], axis=0)
        s_merge_df = s_merge_df[s_merge_df.index.get_level_values(1).isin(s_index)]
        merge_df = pd.concat([n_merge_df, s_merge_df], axis=0)
        merge_df['mean'] -= 273.15
        print('all data shape', merge_df.shape)
        wide_df = merge_df['mean'].unstack(level='year')
        print('reshape data', wide_df.shape, wide_df.index, wide_df.columns)
        wide_df.to_csv(save_path, index_label='Hylak_id')
    else:
        wide_df = pd.read_csv(save_path, header=0, index_col='Hylak_id')
    wide_df.columns = [int(float(col)) for col in wide_df.columns]
    print('load hydrolakes temp')
    return wide_df
