# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime,timedelta,date,time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
import copy

# 录入数据
travel_time_train = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/travel_time_train_data.csv')
volume_train = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/volume_train_data.csv')

test_travel_time = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/test_travel_time_data.csv')
test_volume = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/test_volume_data.csv')

# 分割路径，对

A2_train = travel_time_train[travel_time_train['route'] == 'A-2']
A3_train = travel_time_train[travel_time_train['route'] == 'A-3']
B1_train = travel_time_train[travel_time_train['route'] == 'B-1']
B3_train = travel_time_train[travel_time_train['route'] == 'B-3']
C1_train = travel_time_train[travel_time_train['route'] == 'C-1']
C3_train = travel_time_train[travel_time_train['route'] == 'C-3']

A2_test = test_travel_time[test_travel_time['route'] == 'A-2']
A3_test = test_travel_time[test_travel_time['route'] == 'A-3']
B1_test = test_travel_time[test_travel_time['route'] == 'B-1']
B3_test = test_travel_time[test_travel_time['route'] == 'B-3']
C1_test = test_travel_time[test_travel_time['route'] == 'C-1']
C3_test = test_travel_time[test_travel_time['route'] == 'C-3']

V10_train =  volume_train[volume_train['pair'] == '1-0']
V11_train =  volume_train[volume_train['pair'] == '1-1']
V20_train =  volume_train[volume_train['pair'] == '2-0']
V30_train =  volume_train[volume_train['pair'] == '3-0']
V31_train =  volume_train[volume_train['pair'] == '3-1']

V10_test =  test_volume[test_volume['pair'] == '1-0']
V11_test =  test_volume[test_volume['pair'] == '1-1']
V20_test =  test_volume[test_volume['pair'] == '2-0']
V30_test =  test_volume[test_volume['pair'] == '3-0']
V31_test =  test_volume[test_volume['pair'] == '3-1']

# 选择特征
time_columns = ['avg_travel_time', 'is_true',
                'month', 'day', 'weekday','holiday','timemap',
                'pressure', 'sea_pressure',
                'wind_direction', 'wind_speed', 'temperature',
                'rel_humidity',
                'precipitation',
                #'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1',
                #'last_20min_C3', 'last_20min_V10', 'last_20min_V11', 'last_20min_V20',
                #'last_20min_V30', 'last_20min_V31'
               ]

time_columns2 = ['month', 'day', 'weekday','holiday','timemap',
                'pressure', 'sea_pressure',
                'wind_direction', 'wind_speed', 'temperature',
                'rel_humidity',
                'precipitation',
                #'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1',
                #'last_20min_C3', 'last_20min_V10', 'last_20min_V11', 'last_20min_V20',
                #'last_20min_V30', 'last_20min_V31'
                ]

volume_columns = ['volume', 'is_true',
                  'month', 'day', 'weekday', 'holiday', 'timemap',
                  'pressure', 'sea_pressure',
                  'wind_direction', 'wind_speed', 'temperature',
                  'rel_humidity',
                  'precipitation',
                  #'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1', 'last_20min_C3',
                  #'last_20min_V10', 'last_20min_V11', 'last_20min_V20', 'last_20min_V30',
                  #'last_20min_V31'
                 ]

volume_columns2 = ['month', 'day', 'weekday', 'holiday', 'timemap',
                  'pressure', 'sea_pressure',
                  'wind_direction', 'wind_speed', 'temperature',
                  'rel_humidity',
                  'precipitation',
                  #'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1', 'last_20min_C3',
                  #'last_20min_V10', 'last_20min_V11', 'last_20min_V20', 'last_20min_V30',
                  #'last_20min_V31'
                  ]

A2_train = A2_train[time_columns]
A3_train = A3_train[time_columns]
B1_train = B1_train[time_columns]
B3_train = B3_train[time_columns]
C1_train = C1_train[time_columns]
C3_train = C3_train[time_columns]

A2_test = A2_test[time_columns2]
A3_test = A3_test[time_columns2]
B1_test = B1_test[time_columns2]
B3_test = B3_test[time_columns2]
C1_test = C1_test[time_columns2]
C3_test = C3_test[time_columns2]

V10_train =  V10_train[volume_columns]
V11_train =  V11_train[volume_columns]
V20_train =  V20_train[volume_columns]
V30_train =  V30_train[volume_columns]
V31_train =  V31_train[volume_columns]

V10_test =  V10_test[volume_columns2]
V11_test =  V11_test[volume_columns2]
V20_test =  V20_test[volume_columns2]
V30_test =  V30_test[volume_columns2]
V31_test =  V31_test[volume_columns2]

# 定义评价函数
# 自定义评价函数
def MAPE(preds, dtrain):
    labels = dtrain.get_label()
    return 'MAPE', float(sum(np.fabs((labels - preds) / labels))) / len(labels)

# 自定义评分函数
def MAPE2(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
score = make_scorer(MAPE2, greater_is_better=False)

# 训练预测
# 分割训练集验证集
A2_train_true = A2_train[A2_train['is_true'] == True]
A2_train_false = A2_train[A2_train['is_true'] == False]

X_train, X_val, y_train, y_val = train_test_split(A2_train_true.iloc[:,1:], A2_train_true.iloc[:,0], test_size=0.2, random_state=63)

X_train = pd.concat([X_train, A2_train_false.iloc[:,1:]], axis=0)
y_train = pd.concat([y_train, A2_train_false.iloc[:,0]])

del X_train['is_true']
del X_val['is_true']

# 转化格式
time_train = xgb.DMatrix(X_train, label=y_train)
time_val = xgb.DMatrix(X_val, label=y_val)
time_test = xgb.DMatrix(A2_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}

plst = list(params.items())
num_rounds = 64
watchlist = [(time_train, 'train'), (time_val, 'val')]

A2time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", A2time_model.best_ntree_limit)

A2time_predict = A2time_model.predict(time_test)

# 分割训练集验证集
A3_train_true = A3_train[A3_train['is_true'] == True]
A3_train_false = A3_train[A3_train['is_true'] == False]

X_train, X_val, y_train, y_val = train_test_split(A3_train_true.iloc[:,1:], A3_train_true.iloc[:,0], test_size=0.2, random_state=8)

X_train = pd.concat([X_train, A3_train_false.iloc[:,1:]], axis=0)
y_train = pd.concat([y_train, A3_train_false.iloc[:,0]])

del X_train['is_true']
del X_val['is_true']

# 转化格式
time_train = xgb.DMatrix(X_train, label=y_train)
time_val = xgb.DMatrix(X_val, label=y_val)
time_test = xgb.DMatrix(A3_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}

plst = list(params.items())
num_rounds = 68
watchlist = [(time_train, 'train'), (time_val, 'val')]

A3time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", A2time_model.best_ntree_limit)

A3time_predict = A3time_model.predict(time_test)

# 分割训练集验证集
B1_train_true = B1_train[B1_train['is_true'] == True]
B1_train_false = B1_train[B1_train['is_true'] == False]

X_train, X_val, y_train, y_val = train_test_split(B1_train_true.iloc[:,1:], B1_train_true.iloc[:,0], test_size=0.2, random_state=23)

X_train = pd.concat([X_train, B1_train_false.iloc[:,1:]], axis=0)
y_train = pd.concat([y_train, B1_train_false.iloc[:,0]])

del X_train['is_true']
del X_val['is_true']
# 转化格式
time_train = xgb.DMatrix(X_train, label=y_train)
time_val = xgb.DMatrix(X_val, label=y_val)
time_test = xgb.DMatrix(B1_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}

plst = list(params.items())
num_rounds = 66
watchlist = [(time_train, 'train'), (time_val, 'val')]

B1time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", B1time_model.best_ntree_limit)

B1time_predict = B1time_model.predict(time_test)

# 分割训练集验证集
B3_train_true = B3_train[B3_train['is_true'] == True]
B3_train_false = B3_train[B3_train['is_true'] == False]

X_train, X_val, y_train, y_val = train_test_split(B3_train_true.iloc[:,1:], B3_train_true.iloc[:,0], test_size=0.2, random_state=29)

X_train = pd.concat([X_train, B3_train_false.iloc[:,1:]], axis=0)
y_train = pd.concat([y_train, B3_train_false.iloc[:,0]])

del X_train['is_true']
del X_val['is_true']
# 转化格式
time_train = xgb.DMatrix(X_train, label=y_train)
time_val = xgb.DMatrix(X_val, label=y_val)
time_test = xgb.DMatrix(B3_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}

plst = list(params.items())
num_rounds = 88
watchlist = [(time_train, 'train'), (time_val, 'val')]

B3time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", B3time_model.best_ntree_limit)

B3time_predict = B3time_model.predict(time_test)

# 分割训练集验证集
C1_train_true = C1_train[C1_train['is_true'] == True]
C1_train_false = C1_train[C1_train['is_true'] == False]

X_train, X_val, y_train, y_val = train_test_split(C1_train_true.iloc[:,1:], C1_train_true.iloc[:,0], test_size=0.2, random_state=77)

X_train = pd.concat([X_train, C1_train_false.iloc[:,1:]], axis=0)
y_train = pd.concat([y_train, C1_train_false.iloc[:,0]])

del X_train['is_true']
del X_val['is_true']
# 转化格式
time_train = xgb.DMatrix(X_train, label=y_train)
time_val = xgb.DMatrix(X_val, label=y_val)
time_test = xgb.DMatrix(C1_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}

plst = list(params.items())
num_rounds = 80
watchlist = [(time_train, 'train'), (time_val, 'val')]

C1time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", C1time_model.best_ntree_limit)

C1time_predict = C1time_model.predict(time_test)

# 分割训练集验证集
C3_train_true = C3_train[C3_train['is_true'] == True]
C3_train_false = C3_train[C3_train['is_true'] == False]

X_train, X_val, y_train, y_val = train_test_split(C3_train_true.iloc[:,1:], C3_train_true.iloc[:,0], test_size=0.2, random_state=177)

X_train = pd.concat([X_train, C3_train_false.iloc[:,1:]], axis=0)
y_train = pd.concat([y_train, C3_train_false.iloc[:,0]])

del X_train['is_true']
del X_val['is_true']
# 转化格式
time_train = xgb.DMatrix(X_train, label=y_train)
time_val = xgb.DMatrix(X_val, label=y_val)
time_test = xgb.DMatrix(C3_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}

plst = list(params.items())
num_rounds = 70
watchlist = [(time_train, 'train'), (time_val, 'val')]

C3time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", C3time_model.best_ntree_limit)

C3time_predict = C3time_model.predict(time_test)

# 分割训练集验证集
X_train, X_val, y_train, y_val = train_test_split(V10_train.iloc[:,1:], V10_train.iloc[:,0], test_size=0.2, random_state=5)

# 转化格式
volume_train = xgb.DMatrix(X_train, label=y_train)
volume_val = xgb.DMatrix(X_val, label=y_val)
volume_test = xgb.DMatrix(V10_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}


plst = list(params.items())
num_rounds = 55
watchlist = [(volume_train, 'train'), (volume_val, 'val')]

V10volume_model = xgb.train(plst, volume_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", V10volume_model.best_ntree_limit)

V10volume_predict = V10volume_model.predict(volume_test)


# 分割训练集验证集
X_train, X_val, y_train, y_val = train_test_split(V11_train.iloc[:,1:], V11_train.iloc[:,0], test_size=0.2, random_state=5)

# 转化格式
volume_train = xgb.DMatrix(X_train, label=y_train)
volume_val = xgb.DMatrix(X_val, label=y_val)
volume_test = xgb.DMatrix(V11_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}


plst = list(params.items())
num_rounds = 1000
watchlist = [(volume_train, 'train'), (volume_val, 'val')]

V11volume_model = xgb.train(plst, volume_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", V11volume_model.best_ntree_limit)

V11volume_predict = V11volume_model.predict(volume_test)

# 分割训练集验证集
X_train, X_val, y_train, y_val = train_test_split(V20_train.iloc[:,1:], V20_train.iloc[:,0], test_size=0.2, random_state=5)

# 转化格式
volume_train = xgb.DMatrix(X_train, label=y_train)
volume_val = xgb.DMatrix(X_val, label=y_val)
volume_test = xgb.DMatrix(V20_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}


plst = list(params.items())
num_rounds = 60
watchlist = [(volume_train, 'train'), (volume_val, 'val')]

V20volume_model = xgb.train(plst, volume_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", V20volume_model.best_ntree_limit)

V20volume_predict = V20volume_model.predict(volume_test)

# 分割训练集验证集
X_train, X_val, y_train, y_val = train_test_split(V30_train.iloc[:,1:], V30_train.iloc[:,0], test_size=0.2, random_state=5)

# 转化格式
volume_train = xgb.DMatrix(X_train, label=y_train)
volume_val = xgb.DMatrix(X_val, label=y_val)
volume_test = xgb.DMatrix(V30_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}


plst = list(params.items())
num_rounds = 85
watchlist = [(volume_train, 'train'), (volume_val, 'val')]

V30volume_model = xgb.train(plst, volume_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", V30volume_model.best_ntree_limit)

V30volume_predict = V30volume_model.predict(volume_test)

# 分割训练集验证集
X_train, X_val, y_train, y_val = train_test_split(V31_train.iloc[:,1:], V31_train.iloc[:,0], test_size=0.2, random_state=5)

# 转化格式
volume_train = xgb.DMatrix(X_train, label=y_train)
volume_val = xgb.DMatrix(X_val, label=y_val)
volume_test = xgb.DMatrix(V31_test)

#参数设置
params = {
    'booster':'gbtree',      #gbtree和gblinear
    'objective':'reg:linear',
    'eta':0.04,             #为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3,通常最后设置eta为0.01~0.2
    #'gamma':0.3,            #模型在默认情况下，对于一个节点的划分只有在其loss function得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，
    'max_depth':5,          #树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合,建议通过交叉验证（xgb.cv ) 进行调参,通常取值：3-10
    #'min_child_weight':2,   #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    #'max_delta_step':2,     #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    'subsample':1,          #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    'colsample_bytree':1,   #在建立树时对特征随机采样的比例。缺省值为1
    #'alpha':5,              #L1 正则的惩罚系数,当数据维度极高时可以使用，使得算法运行更快。
    #'lambda':40,           #L2 正则的惩罚系数,用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    'seed':2017,
}


plst = list(params.items())
num_rounds = 55
watchlist = [(volume_train, 'train'), (volume_val, 'val')]

V31volume_model = xgb.train(plst, volume_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", V31volume_model.best_ntree_limit)

V31volume_predict = V31volume_model.predict(volume_test)

# 整合及输出
submission_travel_time = test_travel_time[['intersection_id','tollgate_id','time_window']]
predict_result = np.concatenate([A2time_predict,A3time_predict,B1time_predict,B3time_predict,C1time_predict,C3time_predict], axis=0)
submission_travel_time['avg_travel_time'] = predict_result

submission_travel_time.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.5/xgboost_original/travel_time_submission.csv', index=False)