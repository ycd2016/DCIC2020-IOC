# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:36:40 2020

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

from scipy import stats

trn_path = config.train_dir
test_path = config.test_dir

percent1 = 0.6
percent2 = 0.4
length = []
sentence = []


def get_data(path, get_type=True):
    features = []
    for file in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        df['day'] = df['time'].apply(lambda x: int(x[0:4]))
        df['month'] = df['time'].apply(lambda x: int(x[0:2]))
        df['hour'] = df['time'].apply(lambda x: int(x[5:7]))
        df['minute'] = df['time'].apply(lambda x: int(x[8:10]))
        df['seconds'] = df['time'].apply(lambda x: int(x[11:13]))
        df[
            'time_transform'
        ] = df['day'] * 24 + df['hour'] + df['minute'] / 60 + df['seconds'] / 3600
        df['dis'] = np.sqrt((df['lat'] - 0) ** 2 + (df['lon'] - 0) ** 2)
        tmp_arr = df['lon'].value_counts().values
        tmp_arr2 = df[df['速度'] != 0]['速度'].values
        tmp_arr3 = df[df['速度'] != 0]['方向'].values
        tmp = stats.mode(tmp_arr2)[0]

        if len(tmp) != 0:
            tmp2 = np.percentile(tmp_arr2, 80),
            tmp2 = tmp2[0]
            tmp3 = np.percentile(tmp_arr2, 20),
            tmp3 = tmp3[0]
            tmp4 = tmp2 - tmp3,
            tmp4 = tmp4[0]
            tmp = tmp[0]
            tmp_feature2 = [tmp2, tmp3, tmp4, tmp, np.median(tmp_arr3), ]
        else:
            tmp4 = np.nan
            tmp2 = np.nan
            tmp = np.nan
            tmp3 = np.nan
            tmp_feature2 = [tmp2, tmp3, tmp4, tmp, np.nan, ]

        tmp_feature = tmp_feature2 + [
            df['lat'].mean(), df['lat'].quantile(0.9), df['lat'].quantile(0.1),
            df['lat'].max(), df['lat'].min(), df['lat'].mode().mean(),
            df['lat'].median(), df['lat'].quantile(percent2),
            df['lat'].quantile(percent1), df['lat'].quantile(percent1) -
            df['lat'].quantile(percent2), df['lat'].nunique(), df['lon'].mean(
            ), df['lon'].quantile(0.9), df['lon'].quantile(0.1), df['lon'].max(
            ), df['lon'].min(), df['lon'].mode().mean(), df['lon'].median(
            ), df['lon'].quantile(percent2), df['lon'].quantile(percent1),
            df['lon'].quantile(percent1) - df['lon'].quantile(percent2),
            df['lon'].nunique(), tmp_arr[0], np.mean(tmp_arr), np.std(tmp_arr),
            df['dis'].mean(), df['dis'].max(), df['dis'].min(), df['dis'].mode(
            ).mean(), df['dis'].median(), df['速度'].max(), df['速度'].std(),
            df['速度'].quantile(0.4), df['速度'].quantile(0.6), df['速度'].quantile(
                0.6) - df['速度'].quantile(0.4), df['速度'].median(), df['方向'].max(
                ), df['方向'].median(), df['方向'].quantile(0.4), df['方向'].quantile(
                    0.6), df['方向'].quantile(0.6) - df['方向'].quantile(0.4),
            (df['lon'].max() - df['lon'].min()) /
            (df['lat'].max() - df['lat'].min() + 0.0001), file
        ]

        if get_type:
            tmp_feature.append(df['type'][0])
        features.append(tmp_feature)
    df = pd.DataFrame(features)
    if get_type:
        df = df.rename(columns={len(features[0]) - 1: 'label'})
        df = df.rename(columns={len(features[0]) - 2: 'filename'})
        label_dict = {'拖网': 0, '刺网': 1, '围网': 2}
        df['label'] = df['label'].map(label_dict)
    else:
        df = df.rename(columns={len(features[0]) - 1: 'filename'})

    return df


df_train = get_data(trn_path)
df_test = get_data(test_path, get_type=False)

df_train['id'] = df_train['filename'].apply(lambda x: int(x.split('.')[0]))
df_test['filename'] = df_test['filename'].apply(lambda x: int(x.split('.')[0]))
df_train.to_csv('feature_train.csv', index=None)
df_test.to_csv('feature_test.csv', index=None)
