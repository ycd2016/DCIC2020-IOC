# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:45:02 2020

@author: hcb
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import warnings
from config import config
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
        df_list.append(df)
    df = pd.concat(df_list)
    return df


df_train = get_data(trn_path)
df_train['dis'] = np.sqrt((df_train['lat'] - 0) ** 2 +
                          (df_train['lon'] - 0) ** 2)
df_test = get_data(test_path)
df_test['dis'] = np.sqrt((df_test['lat'] - 0) ** 2 + (df_test['lon'] - 0) ** 2)


def transform_day(df):
    df['hour'] = df['time'].apply(lambda x: int(x[5:7]))
    df['minute'] = df['time'].apply(lambda x: int(x[8:10]))
    df['seconds'] = df['time'].apply(lambda x: int(x[11:13]))
    df['time_transform'] = df['hour'] + df['minute'] / 60 + df['seconds'] / 3600
    return df


df_train = transform_day(df_train)
df_test = transform_day(df_test)

train = df_train[['渔船ID', 'type']].drop_duplicates()
type_dict = {'拖网': 0, '刺网': 1, '围网': 2}
train['type'] = train['type'].map(type_dict)
test = df_test[['渔船ID']].drop_duplicates()

feature_train = pd.read_csv('feature_train.csv')
feature_train = feature_train.rename(columns={'id': '渔船ID'})
feature_train.drop(['filename', 'label'], axis=1, inplace=True)
train = train.merge(feature_train, on='渔船ID', how='left')
del feature_train

feature_test = pd.read_csv('feature_test.csv')
feature_test = feature_test.rename(columns={'filename': '渔船ID'})
test = test.merge(feature_test, on='渔船ID', how='left')
del feature_test

feature_train = pd.read_csv('train2.csv')
train = train.merge(feature_train, on='渔船ID', how='left')
del feature_train

feature_test = pd.read_csv('test2.csv')
test = test.merge(feature_test, on='渔船ID', how='left')
del feature_test


def get_f3(df, df2):

    df3 = df2[df2['速度'] > 0]

    tmp_df = df3.groupby('渔船ID')['速度'].agg({'sudu_min': 'min'}).reset_index()
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['方向'].agg(
        {'fx_unique': 'nunique'}).reset_index()
    df = df.merge(tmp_df, on='渔船ID', how='left')

    return df


train = get_f3(train, df_train)
test = get_f3(test, df_test)


def mode_mean(x):
    return x.mode().mean()


def get_f1(df, df2):
    df3 = df2[df2['速度'] < 1]

    tmp_df = df3.groupby('渔船ID')['lat'].agg(
        {'lat_median': 'median',
         'lat_mode': mode_mean}).reset_index()
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['lon'].agg(
        {'lon_median': 'median',
         'lon_mode': mode_mean}).reset_index()
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['lat'].agg({
        'lat_median2': 'median',
        'latmean2': 'mean',
        'latm3': mode_mean
    }).reset_index()  # , '方向skew':'skew'
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['lat'].agg(lambda x: x.quantile(0.8)).to_dict()
    df['x_0.8'] = df['渔船ID'].map(tmp_df)

    tmp_df = df3.groupby('渔船ID')['lat'].agg(lambda x: x.quantile(0.7)).to_dict()
    df['x_0.7'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['lat'].agg(lambda x: x.quantile(0.9)).to_dict()
    df['x_0.9'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['lat'].agg(lambda x: x.quantile(0.3)).to_dict()
    df['x_0.3'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['lat'].agg(lambda x: x.quantile(0.1)).to_dict()
    df['x_0.1'] = df['渔船ID'].map(tmp_df)

    tmp_df = df3.groupby('渔船ID')['lat'].agg(lambda x: x.quantile(0.2)).to_dict()
    df['x_0.2'] = df['渔船ID'].map(tmp_df)

    tmp_df = df3.groupby('渔船ID')['lon'].agg(lambda x: x.quantile(0.8)).to_dict()
    df['y_0.8'] = df['渔船ID'].map(tmp_df)

    tmp_df = df3.groupby('渔船ID')['lon'].agg(lambda x: x.quantile(0.7)).to_dict()
    df['y_0.7'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['lon'].agg(lambda x: x.quantile(0.9)).to_dict()
    df['y_0.9'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['lon'].agg(lambda x: x.quantile(0.3)).to_dict()
    df['y_0.3'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['lon'].agg(lambda x: x.quantile(0.1)).to_dict()
    df['y_0.1'] = df['渔船ID'].map(tmp_df)

    tmp_df = df3.groupby('渔船ID')['lon'].agg(lambda x: x.quantile(0.2)).to_dict()
    df['y_0.2'] = df['渔船ID'].map(tmp_df)

    tmp_df = df3.groupby('渔船ID')['dis'].agg({
        'dis_m1': 'median',
        'dis_m2': 'mean',
        'dis_m3': mode_mean
    }).reset_index()  # , '方向skew':'skew'
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['dis'].agg(lambda x: x.quantile(0.8)).to_dict()
    df['dis_0.8'] = df['渔船ID'].map(tmp_df)

    tmp_df = df3.groupby('渔船ID')['dis'].agg(lambda x: x.quantile(0.2)).to_dict()
    df['dis_0.2'] = df['渔船ID'].map(tmp_df)

    df3 = df2[df2['速度'] > 0]

    tmp_df = df3.groupby('渔船ID')['lat'].agg(
        {'lat_m1': 'mean',
         'lat_m2': 'median',
         'lat_std': 'std'}).reset_index()
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['lon'].agg(
        {'lon_m1': 'mean',
         'lon_m2': 'median',
         'lon_std': 'std'}).reset_index()  # , '方向skew':'skew'
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['速度'].agg(lambda x: x.quantile(0.85)).to_dict()
    df['速度_0.85'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['速度'].agg(lambda x: x.quantile(0.7)).to_dict()
    df['速度_0.7'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['速度'].agg(lambda x: x.quantile(0.15)).to_dict()
    df['速度_0.15'] = df['渔船ID'].map(tmp_df)
    tmp_df = df3.groupby('渔船ID')['速度'].agg(lambda x: x.quantile(0.3)).to_dict()
    df['速度_0.3'] = df['渔船ID'].map(tmp_df)
    return df


train = get_f1(train, df_train)
test = get_f1(test, df_test)


def get_f2(df, df2):
    df3 = df2[df2['速度'] < 1]

    tmp_df = df3.groupby('渔船ID')['time_transform'].agg({'count'}).reset_index(
    )  # , '方向skew':'skew'
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['lat'].agg({'nunique'}).reset_index(
    )  # , '方向skew':'skew'
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['lon'].agg({'nunique'}).reset_index(
    )  # , '方向skew':'skew'
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df3.groupby('渔船ID')['渔船ID'].agg({'count'}).reset_index(
    )  # , '方向skew':'skew'
    tmp_df = tmp_df.rename(columns={'count': 'id_count'})
    df = df.merge(tmp_df, on='渔船ID', how='left')

    tmp_df = df2.groupby('渔船ID')['渔船ID'].agg({'count'}).reset_index(
    )  # , '方向skew':'skew'
    tmp_df = tmp_df.rename(columns={'count': 'id_count2'})
    df = df.merge(tmp_df, on='渔船ID', how='left')

    df['propotion_v=0'] = df['id_count'] / df['id_count2']
    df.drop(['id_count', 'id_count2'], axis=1, inplace=True)
    return df


train = get_f2(train, df_train)
test = get_f2(test, df_test)


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', scores, True


folds = KFold(n_splits=5, shuffle=True, random_state=1996)
oof_lgb = np.zeros((len(train), 3))

col = [tmp_col for tmp_col in train.columns
       if tmp_col not in ['type', '渔船ID', 'label']]
X_train = train[col].values
y_train = train['type'].values
prediction = np.zeros((len(test), 3))
print(len(col))
print(X_train.shape)
param = {
    'learning_rate': 0.06,
    'boosting_type': 'gbdt',
    'objective': 'multiclassova',
    'metric': 'None',
    'num_leaves': 35,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'seed': 1,
    'bagging_seed': 1,
    'feature_fraction_seed': 7,
    'min_data_in_leaf': 20,
    'num_class': 3,
    'nthread': 4,
    'verbose': -1,
}

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 3000
    clf = lgb.train(param, trn_data, num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=500,
                    early_stopping_rounds=300,
                    feval=f1_score_eval)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx],
                                   num_iteration=clf.best_iteration)
    prediction += clf.predict(test[col].values,
                              num_iteration=clf.best_iteration)

oof_lgb_final = np.argmax(oof_lgb, axis=1)
print(f1_score(y_train, oof_lgb_final, average='macro'))

result = []
for i in range(3):
    tp = sum((y_train == i) & (oof_lgb_final == i))
    recall = tp / sum(y_train == i)
    precision = tp / sum(oof_lgb_final == i)
    fscore = 2 * recall * precision / (recall + precision)
    print('%d° recall: %.3f, precision: %.3f, fscore: %.3f' %
          (i, recall, precision, fscore))

prediction = prediction / 5
pred2 = pd.read_csv('../qyxs/qyxs_prob.csv')
pred2 = pred2.rename(columns={'ID': '渔船ID'})
pred3 = pd.read_csv('../Grand_Rookie/rookie_prob.csv')
tmp_df = pd.DataFrame()
tmp_df['渔船ID'] = test['渔船ID']
tmp_df = tmp_df.merge(pred2, how='left', on='渔船ID')
tmp_df = tmp_df.merge(pred3, how='left', on='渔船ID')
prediction2 = tmp_df[['qyxs_prob_拖网', 'qyxs_prob_刺网', 'qyxs_prob_围网']].values
prediction3 = tmp_df[['rookie_prob_拖网', 'rookie_prob_刺网',
                      'rookie_prob_围网']].values
prediction = 0.5 * prediction2 + 0.3 * prediction + 0.2 * prediction3

pred_label = np.argmax(prediction, axis=1)
label_dict = {0: '拖网', 1: '刺网', 2: '围网'}
df_pred = pd.DataFrame()
df_pred['渔船ID'] = test['渔船ID']
df_pred['label'] = pred_label
df_pred['label'] = df_pred['label'].map(label_dict)
df_pred.to_csv(config.save_path, index=None, header=False)
