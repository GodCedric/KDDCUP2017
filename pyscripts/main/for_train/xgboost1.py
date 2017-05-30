# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
XGBoost for prediction
Created on 2017/05/17
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time


# 录入数据
travel_time_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/travel_time_train_data.csv')
volume_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/volume_train_data.csv')
travel_time_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/test_travel_time_data.csv')
volume_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/test_volume_data.csv')


# 分割训练集和验证集
test_date = ['2016-10-18', '2016-10-19', '2016-10-20', '2016-10-21', '2016-10-22', '2016-10-23', '2016-10-24']
def isin(x):
    if x in test_date:
        return 1
    else:
        return 0

# 平均时间
travel_time_train_data['is_test'] = travel_time_train_data['date'].map(isin)
travel_time_train = travel_time_train_data[travel_time_train_data['is_test'] == 0]
travel_time_val = travel_time_train_data[travel_time_train_data['is_test'] == 1]
# 流量
volume_train_data['is_test'] = volume_train_data['date'].map(isin)
volume_train = volume_train_data[volume_train_data['is_test'] == 0]
volume_val = volume_train_data[volume_train_data['is_test'] == 1]

# 分离标签和特征
Y_time = travel_time_train.avg_travel_time
X_time = travel_time_train.drop(['avg_travel_time'], axis=1)
val_Y_time = travel_time_val.avg_travel_time
val_X_time = travel_time_val.drop(['avg_travel_time'], axis=1)

Y_volume = volume_train.volume
X_volume = volume_train.drop(['volume'], axis=1)
val_Y_volume = volume_val.volume
val_X_volume = volume_val.drop(['volume'], axis=1)


# 选择特征
travel_time_features = ['route', 'weekday', 'timemap',\
                        'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                        #'last_20min',\
                        'SSD',\
                        'is_workday'
                       ]

volume_features = ['pair', 'weekday', 'timemap',\
                   'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                   #'last_20min',\
                   'SSD',\
                   'is_workday'
                  ]

"""
# 选择特征
travel_time_features = ['A-2', 'A-3', 'B-1', 'B-3', 'C-1', 'C-3',
                        'weekday__0', 'weekday__1', 'weekday__2', 'weekday__3', 'weekday__4', 'weekday__5', 'weekday__6',
                        'workday__1', 'workday__2', 'workday__3',
                        'hour__0', 'hour__1', 'hour__2','hour__3', 'hour__4', 'hour__5', 'hour__6', 'hour__7', 'hour__8',
                        'hour__9', 'hour__10', 'hour__11', 'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16',
                        'hour__17', 'hour__18', 'hour__19', 'hour__20',
                        'hour__21', 'hour__22', 'hour__23', 'minute__0', 'minute__20', 'minute__40',
                        'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                        #'last_20min',\
                        'SSD'
                       ]

volume_features = ['1-0', '1-1', '2-0', '3-0', '3-1',
                   'weekday__1', 'weekday__2', 'weekday__3', 'weekday__4', 'weekday__5', 'weekday__6', 'workday__1', 'workday__2',
                   'hour__0', 'hour__1', 'hour__2','hour__3', 'hour__4', 'hour__5', 'hour__6', 'hour__7', 'hour__8',
                   'hour__9', 'hour__10', 'hour__11', 'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16',
                   'hour__17', 'hour__18', 'hour__19', 'hour__20',
                   'hour__21', 'hour__22', 'hour__23', 'minute__0', 'minute__20', 'minute__40',
                   'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                   #'last_20min',\
                   'SSD'
                  ]
"""

X_time = X_time[travel_time_features]
val_X_time = val_X_time[travel_time_features]
travel_time_submission = travel_time_test_data[['intersection_id', 'tollgate_id', 'time_window']]
travel_time_test_data = travel_time_test_data[travel_time_features]

X_volume = X_volume[volume_features]
val_X_volume = val_X_volume[volume_features]
volume_submission = volume_test_data[['tollgate_id', 'time_window', 'direction']]
volume_test_data = volume_test_data[volume_features]


# 字符串特征转换数值
X_time['route'] = pd.factorize(X_time['route'])[0]
val_X_time['route'] = pd.factorize(val_X_time['route'])[0]
travel_time_test_data['route'] = pd.factorize(travel_time_test_data['route'])[0]
X_volume['pair'] = pd.factorize(X_volume['pair'])[0]
val_X_volume['pair'] = pd.factorize(val_X_volume['pair'])[0]
volume_test_data['pair'] = pd.factorize(volume_test_data['pair'])[0]

# 转化格式
time_train = xgb.DMatrix(X_time, label=Y_time)
time_val = xgb.DMatrix(val_X_time, label=val_Y_time)
time_test = xgb.DMatrix(travel_time_test_data)

volume_train = xgb.DMatrix(X_volume, label=Y_volume)
volume_val = xgb.DMatrix(val_X_volume, label=val_Y_volume)
volume_test = xgb.DMatrix(volume_test_data)

# 参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.08,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':0.8,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}

# 训练
# 自定义评价函数
def MAPE(preds, dtrain):
    labels = dtrain.get_label()
    return 'MAPE', float(sum(np.fabs((labels - preds) / labels))) / len(labels)
plst = list(params.items())
num_rounds = 1000
watchlist = [(time_train, 'train'), (time_val, 'val')]

time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", time_model.best_ntree_limit)

# 预测
time_predict = time_model.predict(time_test)
# 输出
travel_time_submission['avg_travel_time'] = time_predict
travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.0/xgboost_liu/travel_time_submission.csv', index=False)


params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.05,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':3,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':0,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':0,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':0.8,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':0,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':120,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2016,
}

plst = list(params.items())
num_rounds = 1000
watchlist = [(volume_train, 'train'), (volume_val, 'val')]

volume_model = xgb.train(plst, volume_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", volume_model.best_ntree_limit )

volume_predict = volume_model.predict(volume_test, ntree_limit=volume_model.best_ntree_limit)

volume_submission['volume'] = volume_predict
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.0/xgboost_liu/volume_submission.csv', index=False)