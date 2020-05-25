# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:33:02 2020

@author: Grand Rookie
"""
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from config import config
warnings.filterwarnings('ignore')


def get_data(path):
    df_list = []
    for file in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    df = pd.concat(df_list)
    return df


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


def transform_degree(x):
    if x > 180:
        return 360 - x
    else:
        return x


def split_degree(degree):
    if np.abs(degree) <= 45:
        return 'small'
    elif np.abs(degree) <= 90 and np.abs(degree) > 45:
        return 'medium'
    elif np.abs(degree) <= 135 and np.abs(degree) > 90:
        return 'large'
    elif np.abs(degree) <= 180 and np.abs(degree) > 135:
        return 'reverse'
    else:
        return 'other'


def geohash_encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)


def get_latlng(df, k):
    x = df['lon'].values
    y = df['lat'].values
    tmp_df = pd.DataFrame()
    tmp_df['lng'] = x
    tmp_df['lat'] = y
    tmp_df['code'] = tmp_df[[
        'lng', 'lat'
    ]].apply(lambda x: geohash_encode(x['lat'], x['lng'],
                                      precision=k),
             axis=1)
    code = tmp_df['code'].values
    return code


print('读取数据...')
TRAIN_PATH = config.train_dir
train_df = get_data(TRAIN_PATH)

TEST_PATH = config.test_dir
test_df = get_data(TEST_PATH)

print('数据处理...')
data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
data = data.sort_values(by=['渔船ID', 'time'])
data = data[(data['lat'] > 0) & (data['lon'] > 0)]
data.reset_index(inplace=True, drop=True)
data['destination'] = data['lat'].apply(lambda x: round(
    x, 3)).map(str) + '_' + data['lon'].apply(lambda x: round(x, 3)).map(str)
data['速度_type'] = data['速度'].apply(lambda x: split_speed(x))
data['方向_type'] = data['速度'].apply(lambda x: split_degree(transform_degree(x)))
data['geohash7'] = get_latlng(data, 7)

print('特征工程...')
features = data[['渔船ID', 'type']].drop_duplicates()

for col in ['lat', 'lon', 'geohash7']:
    df = data.copy()
    df[col] = df[col].astype(str)
    enc_vec = TfidfVectorizer()
    group_df = df.groupby(['渔船ID']).apply(lambda x:
                                          x[col].tolist()).reset_index()
    group_df.columns = ['渔船ID', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ' '.join(x))
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_{}_{}'.format(col, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    features = features.merge(group_df, on=['渔船ID'], how='left')
    del features['list']

for col in ['lat', 'lon']:
    group_df = data.groupby(['渔船ID'])[col].agg({
        col + '_mean': 'mean',
        col + '_max': 'max',
        col + '_min': 'min',
        col + '_nuniq': 'nunique',
        col + '_q1': lambda x: np.quantile(x, 0.10),
        col + '_q2': lambda x: np.quantile(x, 0.20),
        col + '_q3': lambda x: np.quantile(x, 0.30),
        col + '_q4': lambda x: np.quantile(x, 0.40),
        col + '_q5': lambda x: np.quantile(x, 0.50),
        col + '_q6': lambda x: np.quantile(x, 0.60),
        col + '_q7': lambda x: np.quantile(x, 0.70),
        col + '_q8': lambda x: np.quantile(x, 0.80),
        col + '_q9': lambda x: np.quantile(x, 0.90)
    }).reset_index()
    features = features.merge(group_df, on=['渔船ID'], how='left')

group_df = data.groupby(['渔船ID'])['destination'].agg(
    {'destination_cnt': 'count',
     'destination_nuniq': 'nunique'}).reset_index()
group_df['destination_ratio'
        ] = group_df['destination_nuniq'] / group_df['destination_cnt']
group_df['destination_freq'
        ] = group_df['destination_cnt'] / group_df['destination_nuniq']
features = features.merge(group_df, on=['渔船ID'], how='left')

group_df = data[data['速度'] < 1].groupby(['渔船ID'])['destination'].agg({
    'stop_destination_cnt': 'count',
    'stop_destination_nuniq': 'nunique'
}).reset_index()
group_df[
    'stop_destination_ratio'
] = group_df['stop_destination_nuniq'] / group_df['stop_destination_cnt']
group_df[
    'stop_destination_freq'
] = group_df['stop_destination_cnt'] / group_df['stop_destination_nuniq']
features = features.merge(group_df, on=['渔船ID'], how='left')
features['stop_ratio1'
        ] = features['stop_destination_cnt'] / features['destination_cnt']
features['stop_ratio2'
        ] = features['stop_destination_nuniq'] / features['destination_nuniq']

for col in ['速度']:
    group_df = data[data['速度'] > 0].groupby(['渔船ID'])[col].agg({
        col + '_mean': 'mean',
        col + '_max': 'max',
        col + '_min': 'min',
        col + '_nuniq': 'nunique',
        col + '_q1': lambda x: np.quantile(x, 0.10),
        col + '_q2': lambda x: np.quantile(x, 0.20),
        col + '_q3': lambda x: np.quantile(x, 0.30),
        col + '_q4': lambda x: np.quantile(x, 0.40),
        col + '_q5': lambda x: np.quantile(x, 0.50),
        col + '_q6': lambda x: np.quantile(x, 0.60),
        col + '_q7': lambda x: np.quantile(x, 0.70),
        col + '_q8': lambda x: np.quantile(x, 0.80),
        col + '_q9': lambda x: np.quantile(x, 0.90)
    }).reset_index()
    features = features.merge(group_df, on=['渔船ID'], how='left')

for col in ['lat', 'lon']:
    group_df = data[data['速度'] < 1].groupby(['渔船ID'])[col].agg({
        'stop_' + col + '_mean': 'mean',
        'stop_' + col + '_max': 'max',
        'stop_' + col + '_min': 'min',
        'stop_' + col + '_nuniq': 'nunique',
        'stop_' + col + '_q1': lambda x: np.quantile(x, 0.10),
        'stop_' + col + '_q2': lambda x: np.quantile(x, 0.20),
        'stop_' + col + '_q3': lambda x: np.quantile(x, 0.30),
        'stop_' + col + '_q4': lambda x: np.quantile(x, 0.40),
        'stop_' + col + '_q5': lambda x: np.quantile(x, 0.50),
        'stop_' + col + '_q6': lambda x: np.quantile(x, 0.60),
        'stop_' + col + '_q7': lambda x: np.quantile(x, 0.70),
        'stop_' + col + '_q8': lambda x: np.quantile(x, 0.80),
        'stop_' + col + '_q9': lambda x: np.quantile(x, 0.90)
    }).reset_index()
    features = features.merge(group_df, on=['渔船ID'], how='left')

for col in ['方向']:
    group_df = data.groupby(['渔船ID'])[col].agg({
        col + '_mean': 'mean',
        col + '_max': 'max',
        col + '_min': 'min',
        col + '_nuniq': 'nunique'
    }).reset_index()
    features = features.merge(group_df, on=['渔船ID'], how='left')

for col in ['lat', 'lon']:
    group_df = data[data['方向'] == 0].groupby(['渔船ID'])[col].agg({
        '方向0_' + col + '_mean': 'mean',
        '方向0_' + col + '_max': 'max',
        '方向0_' + col + '_min': 'min',
        '方向0_' + col + '_nuniq': 'nunique',
        '方向0_' + col + '_q1': lambda x: np.quantile(x, 0.10),
        '方向0_' + col + '_q2': lambda x: np.quantile(x, 0.20),
        '方向0_' + col + '_q3': lambda x: np.quantile(x, 0.30),
        '方向0_' + col + '_q4': lambda x: np.quantile(x, 0.40),
        '方向0_' + col + '_q5': lambda x: np.quantile(x, 0.50),
        '方向0_' + col + '_q6': lambda x: np.quantile(x, 0.60),
        '方向0_' + col + '_q7': lambda x: np.quantile(x, 0.70),
        '方向0_' + col + '_q8': lambda x: np.quantile(x, 0.80),
        '方向0_' + col + '_q9': lambda x: np.quantile(x, 0.90)
    }).reset_index()
    features = features.merge(group_df, on=['渔船ID'], how='left')

for col in ['lat', 'lon']:
    group_df = data[data['速度_type'] == 'high'].groupby(['渔船ID'])[col].agg({
        '速度_high_' + col + '_mean': 'mean',
        '速度_high_' + col + '_max': 'max',
        '速度_high_' + col + '_min': 'min',
        '速度_high_' + col + '_nuniq': 'nunique',
        '速度_high_' + col + '_q1': lambda x: np.quantile(x, 0.10),
        '速度_high_' + col + '_q2': lambda x: np.quantile(x, 0.20),
        '速度_high_' + col + '_q3': lambda x: np.quantile(x, 0.30),
        '速度_high_' + col + '_q4': lambda x: np.quantile(x, 0.40),
        '速度_high_' + col + '_q5': lambda x: np.quantile(x, 0.50),
        '速度_high_' + col + '_q6': lambda x: np.quantile(x, 0.60),
        '速度_high_' + col + '_q7': lambda x: np.quantile(x, 0.70),
        '速度_high_' + col + '_q8': lambda x: np.quantile(x, 0.80),
        '速度_high_' + col + '_q9': lambda x: np.quantile(x, 0.90)
    }).reset_index()
    features = features.merge(group_df, on=['渔船ID'], how='left')

for col in ['geohash7']:
    df = data[data['速度_type'] == 'low'].copy()
    df[col] = df[col].astype(str)
    enc_vec = TfidfVectorizer()
    group_df = df.groupby(['渔船ID']).apply(lambda x:
                                          x[col].tolist()).reset_index()
    group_df.columns = ['渔船ID', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ' '.join(x))
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['low_speed_svd_{}_{}'.format(col, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    features = features.merge(group_df, on=['渔船ID'], how='left')
    del features['list']


def get_grad_tfidf(df, group_id, group_target, num):
    grad_df = df.groupby(group_id)['lat'].apply(lambda x:
                                                np.gradient(x)).reset_index()
    grad_df['lon'] = df.groupby('渔船ID')['lon'].apply(
        lambda x: np.gradient(x)).reset_index()['lon']
    grad_df['lat'] = grad_df['lat'].apply(lambda x: np.round(x, 4))
    grad_df['lon'] = grad_df['lon'].apply(lambda x: np.round(x, 4))
    grad_df[group_target] = grad_df.apply(
        lambda x: ' '.join(['{}_{}'.format(z[0], z[1])
                            for z in zip(x['lat'], x['lon'])]),
        axis=1)

    tfidf_enc_tmp = TfidfVectorizer()
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(grad_df[group_target])
    svd_tag_tmp = TruncatedSVD(n_components=num, n_iter=20, random_state=1024)
    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    tag_svd_tmp.columns = ['{}_svd_{}'.format(group_target, i)
                           for i in range(num)]
    return pd.concat([grad_df[[group_id]], tag_svd_tmp], axis=1)


tmp = get_grad_tfidf(data, '渔船ID', 'gradiant', 10)
features = features.merge(tmp, on=['渔船ID'], how='left')


def f1_macro(preds, train_data):
    y_true = train_data.label
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    score = f1_score(y_true, preds, average='macro')
    return 'f1_macro', score, True


def LGB_classfication_model(train, target, test, k):
    drop_cols = [f for f in train.columns
                 if 'lat_lon_cnt' in f or '速度_high' in f]
    feats = [f for f in train.columns if f not in ['渔船ID', 'type'] + drop_cols]
    print('Current num of features:', len(feats))
    folds = KFold(n_splits=k, shuffle=True, random_state=2019)
    oof_preds = np.zeros(train.shape[0])
    oof_probs = np.zeros((train.shape[0], 3))
    feature_importance_df = pd.DataFrame()
    offline_score = []
    offline_f1_score = []
    output_preds = []
    for i, (train_index, test_index) in enumerate(folds.split(train)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train[feats].iloc[train_index,:].values, train[feats].iloc[test_index,:].values

        dtrain = lgb.Dataset(train_X, label=train_y)
        dval = lgb.Dataset(test_X, label=test_y)
        parameters = {
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'None',
            'num_leaves': 63,
            'num_class': 3,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'nthread': 8
        }
        lgb_model = lgb.train(parameters, dtrain,
                              num_boost_round=5000,
                              valid_sets=[dval],
                              early_stopping_rounds=100,
                              verbose_eval=100,
                              feval=f1_macro)
        oof_preds[test_index] = np.argmax(
            lgb_model.predict(test_X,
                              num_iteration=lgb_model.best_iteration),
            axis=1)
        oof_probs[test_index] = lgb_model.predict(
            test_X,
            num_iteration=lgb_model.best_iteration)
        offline_score.append(lgb_model.best_score['valid_0']['f1_macro'])
        offline_f1_score.append(f1_score(test_y, oof_preds[test_index],
                                         average='macro'))
        output_preds.append(
            lgb_model.predict(test[feats],
                              num_iteration=lgb_model.best_iteration))
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = lgb_model.feature_importance(
            importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df,
                                           fold_importance_df],
                                          axis=0)
    print('OOF-MEAN-F1 score:%.6f, OOF-STD:%.6f' %
          (np.mean(offline_f1_score), np.std(offline_f1_score)))
    print('OOF-F1 score:%.6f' % (f1_score(target, oof_preds, average='macro')))
    print('feature importance:')
    print(feature_importance_df.groupby(
        ['feature'])['importance'].mean().sort_values(ascending=False).head(5))

    return output_preds, oof_probs, np.mean(offline_f1_score)


print('开始训练...')
train = features[~features['type'].isnull()].copy()
test = features[features['type'].isnull()].copy()
train['type'] = train['type'].replace({'拖网': 0, '围网': 1, '刺网': 2})
target = train['type'].values
print(train.shape, test.shape)
preds, oof_probs, score = LGB_classfication_model(train, target, test, 5)

print('生成结果...')
use_prob = True
PROB_PATH = config.prob_rookie
if use_prob:
    sub_probs = ['rookie_prob_{}'.format(q) for q in ['拖网', '围网', '刺网']]
    prob_df = pd.DataFrame(np.mean(preds, axis=0), columns=sub_probs)
    prob_df['渔船ID'] = test['渔船ID'].values
    prob_df.to_csv(PROB_PATH, index=False, encoding='utf-8')
else:
    sub_df = test[['渔船ID']].copy()
    sub_df['type'] = np.argmax(np.mean(preds, axis=0), axis=1)
    sub_df['type'] = sub_df['type'].replace({0: '拖网', 1: '围网', 2: '刺网'})
    sub_df.to_csv(config.save_path, index=False, encoding='utf-8', header=None)
