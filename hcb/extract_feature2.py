# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:28:58 2020

@author: hcb
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from config import config
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import geohash
warnings.filterwarnings("ignore")

trn_path = config.train_dir
test_path = config.test_dir


def mode_mean(x):
    return x.mode().mean()


def get_data(path):
    df_list = []
    for file in tqdm(sorted(os.listdir(path))):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        df['time_id'] = list(range(len(df)))
        df_list.append(df)
    df = pd.concat(df_list)
    return df


def get_latlng(df, precision=7):
    tmp_df = pd.DataFrame()
    tmp_df['lng'] = df['lon']
    tmp_df['lat'] = df['lat']
    tmp_df['code'] = tmp_df[[
        'lng', 'lat'
    ]].apply(lambda x: geohash.encode(x['lat'], x['lng'],
                                      precision=precision),
             axis=1)
    code = tmp_df['code'].values
    return code


def transform_day(df):
    df['day'] = df['time'].apply(lambda x: int(x[0:4]))
    df['month'] = df['time'].apply(lambda x: int(x[0:2]))
    df['hour'] = df['time'].apply(lambda x: int(x[5:7]))
    df['minute'] = df['time'].apply(lambda x: int(x[8:10]))
    df['seconds'] = df['time'].apply(lambda x: int(x[11:13]))
    df['time_transform'] = (df['month'] * 31 + df['day']) * 24 + df[
        'hour'
    ] + df['minute'] / 60 + df['seconds'] / 3600
    return df


def get_feature(df2, train):
    df = df2.copy()
    df['new_id'] = (df['渔船ID'] + 1) * 10000 + df['time_id']
    tmp_df = df[['渔船ID', 'lat', 'lon', 'time_transform', 'new_id']].copy()
    tmp_df.columns = ['渔船ID', 'x_1', 'y_1', 'time_transform_1', 'new_id']
    tmp_df['new_id'] = tmp_df['new_id'] + 1
    df = df.merge(tmp_df, on=['渔船ID', 'new_id'], how='left')
    df['dis_path'] = np.sqrt((df['x_1'] - df['lat']) ** 2 +
                             (df['y_1'] - df['lon']) ** 2)
    df['slope'] = np.abs((df['y_1'] - df['lon']) /
                         (df['x_1'] - df['lat'] + 0.001))

    df.dropna(inplace=True)
    tmp_df = df.groupby('渔船ID')['dis_path'].agg({
        'max', 'median', 'mean', 'sum'
    }).reset_index()
    tmp_df.columns = ['渔船ID', 'dis_path_max', 'dis_path_median',
                      'dis_path_mean', 'dis_path_sum']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df.groupby('渔船ID')['slope'].agg({
        'max', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', 'slope_max', 'slope_median', 'slope_mean1']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df[df['速度'] > 0]
    tmp_df = tmp_df.groupby('渔船ID')['dis_path'].agg({
        'min', 'std', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', 'dis_path_min2', 'dis_path_std2',
                      'dis_path_median2', 'dis_path_mean']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df.groupby('渔船ID')['slope'].agg({
        'min', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', 'slope_min', 'slope_median2', 'slope_mean2']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df[df['速度'] > 0]
    tmp_df = tmp_df.groupby('渔船ID')['slope'].agg({
        'min', 'std', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', 'slope_min3', 'slope_std3', 'slope_median3',
                      'slope_mean3']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    df['time_delt'] = np.abs(df['time_transform_1'] - df['time_transform'])
    df['dis/time'] = df['dis_path'] / df['time_delt']

    tmp_df = df.groupby('渔船ID')['dis/time'].agg({
        'mean', 'median'
    }).reset_index()
    tmp_df.columns = ['渔船ID', 'dis/time_mean', 'dis/time_median']
    train = train.merge(tmp_df, on='渔船ID', how='left')
    return train


def get_feature2(df2, train):
    df = df2.copy()
    df['new_id'] = (df['渔船ID'] + 1) * 10000 + df['time_id']
    tmp_df = df[['渔船ID', '方向', '速度', 'new_id']].copy()
    tmp_df.columns = ['渔船ID', '方向_1', '速度_1', 'new_id']
    tmp_df['new_id'] = tmp_df['new_id'] + 1
    df = df.merge(tmp_df, on=['渔船ID', 'new_id'], how='left')
    df['方向_delt'] = np.abs(df['方向_1'] - df['方向'])
    df['速度_delt'] = np.abs(df['速度_1'] - df['速度'])
    df.dropna(inplace=True)

    tmp_df = df.groupby('渔船ID')['方向_delt'].agg({
        'max', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', '方向_delt_mmax', '方向_delt_median', '方向_delt_mean']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df[df['速度'] > 0]
    tmp_df = df.groupby('渔船ID')['方向_delt'].agg({
        'min', 'std', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', '方向_delt_min2', '方向_delt_std2',
                      '方向_delt_median2', '方向_delt_mean2']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df[df['速度'] > 0]
    tmp_df = tmp_df.groupby('渔船ID')['方向_delt'].agg({
        'min', 'std', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', '方向_delt_min3', '方向_delt_std3',
                      '方向_delt_median3', '方向_delt_mean3']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df.groupby('渔船ID')['速度_delt'].agg({
        'max', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', '速度_delt_max', '速度_delt_median', '速度_delt_mean']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df[df['速度'] > 0]
    tmp_df = df.groupby('渔船ID')['速度_delt'].agg({
        'min', 'std', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', '速度_delt_min2', '速度_delt_std2',
                      '速度_delt_median2', '速度_delt_mean2']
    train = train.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df[df['速度'] > 0]
    tmp_df = tmp_df.groupby('渔船ID')['速度_delt'].agg({
        'min', 'std', 'median', 'mean'
    }).reset_index()
    tmp_df.columns = ['渔船ID', '速度_delt_min3', '速度_delt_std3',
                      '速度_delt_median3', '速度_delt_mean3']
    train = train.merge(tmp_df, on='渔船ID', how='left')
    return train


df_train = get_data(trn_path)
train_ = df_train[['渔船ID', 'type']].drop_duplicates()
df_train = transform_day(df_train)
train_ = get_feature(df_train, train_)
train_ = get_feature2(df_train, train_)
train_.drop(['type', 'slope_mean1', 'slope_mean2'], axis=1, inplace=True)

df_test = get_data(test_path)
test = df_test[['渔船ID']].drop_duplicates()
df_test = transform_day(df_test)
test = get_feature(df_test, test)
test = get_feature2(df_test, test)
test.drop(['slope_mean1', 'slope_mean2'], axis=1, inplace=True)

print('begin tfidf')
data = pd.concat((df_train, df_test))
data['destination'] = data['lat'].map(str) + '_' + data['lon'].map(str)

enc_vec = TfidfVectorizer()
group_df = data.groupby(['渔船ID'])['destination'].agg({
    lambda x: list(x)
}).reset_index()

group_df.columns = ['渔船ID', 'destination']
group_df['destination'] = group_df['destination'].apply(lambda x: ' '.join(x))
tfidf_vec = enc_vec.fit_transform(group_df['destination'])
svd_enc = TruncatedSVD(n_components=30, n_iter=20, random_state=1996)
vec_svd = svd_enc.fit_transform(tfidf_vec)
vec_svd = pd.DataFrame(vec_svd)
vec_svd.columns = ['svd_{}_{}'.format('destination', i) for i in range(30)]
group_df = pd.concat([group_df, vec_svd], axis=1)
train_ = train_.merge(group_df, on=['渔船ID'], how='left')
del train_['destination']

test = test.merge(group_df, on=['渔船ID'], how='left')
del test['destination']

data = pd.concat((df_train, df_test))
mode_df = data.groupby(['渔船ID', 'lat',
                        'lon'])['time'].agg({'count'}).reset_index()
mode_df = mode_df.rename(columns={'count': 'mode_count'})
mode_df['rank'] = mode_df.groupby('渔船ID')['mode_count'].rank(method='first',
                                                             ascending=False)

for i in range(1, 4):
    tmp_df = mode_df[mode_df['rank'] == i]
    del tmp_df['rank']
    tmp_df.columns = ['渔船ID', 'rank{}_mode_lat'.format(i),
                      'rank{}_mode_lon'.format(i), 'rank{}_mode_cnt'.format(i)]
    train_ = train_.merge(tmp_df, on='渔船ID', how='left')
    test = test.merge(tmp_df, on='渔船ID', how='left')


def split_speed(speed):
    if speed <= 4:
        return 'low'
    elif speed <= 10 and speed > 4:
        return 'median-low'
    elif speed <= 18 and speed > 10:
        return 'median'
    elif speed <= 50 and speed > 18:
        return 'high'
    else:
        return 'very-high'


tmp_df = data.groupby('渔船ID')['渔船ID'].agg({'count'}).reset_index(
)  # , '方向skew':'skew'
tmp_df = tmp_df.rename(columns={'count': 'id_count'})
train_ = train_.merge(tmp_df, on='渔船ID', how='left')
test = test.merge(tmp_df, on='渔船ID', how='left')

data['速度_type'] = data['速度'].apply(lambda x: split_speed(x))
group_df = data.groupby(['渔船ID', '速度_type']).size().unstack().fillna(0)
group_df.columns = ['速度_' + f + '_cnt' for f in group_df.columns]
group_df.reset_index(inplace=True)
train_ = train_.merge(group_df, on=['渔船ID'], how='left')
test = test.merge(group_df, on=['渔船ID'], how='left')
for col in group_df.columns:
    if col not in ['渔船ID']:
        train_[col.replace('cnt', 'ratio')] = train_[col] / train_['id_count']
        test[col.replace('cnt', 'ratio')] = test[col] / test['id_count']
train_.drop('id_count', axis=1, inplace=True)
test.drop('id_count', axis=1, inplace=True)

countvec = CountVectorizer()
data = pd.concat((df_train, df_test))

code = get_latlng(data)
data['destination'] = code

group_df = data.groupby(['渔船ID'])['destination'].agg({
    lambda x: list(x)
}).reset_index()
group_df.columns = ['渔船ID', 'destination']
group_df['destination'] = group_df['destination'].apply(lambda x: ' '.join(x))

count_vec_tmp = countvec.fit_transform(group_df['destination'])
svd_tmp = TruncatedSVD(n_components=30, n_iter=20, random_state=1996)
svd_tmp = svd_tmp.fit_transform(count_vec_tmp)
svd_tmp = pd.DataFrame(svd_tmp)
svd_tmp.columns = ['{}_countvec_{}'.format('destination', i)
                   for i in range(30)]
group_df = pd.concat([group_df, svd_tmp], axis=1)

train_ = train_.merge(group_df, on=['渔船ID'], how='left')
del train_['destination']

test = test.merge(group_df, on=['渔船ID'], how='left')
del test['destination']

train_.to_csv('train2.csv', index=None)
test.to_csv('test2.csv', index=None)
