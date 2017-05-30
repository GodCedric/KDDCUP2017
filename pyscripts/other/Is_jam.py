# -*- coding: utf-8 -*-
# !/user/bin/evn python

import pandas as pd


path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/trajectories(table 5)_training.csv'
file = pd.read_csv(path)
file['Is_jam'] = 0
column = ['intersection_id','tollgate_id', 'vehicle_id', 'starting_time', 'travel_seq', 'travel_time', 'Is_jam']
# 将原始数据集划分到每个pair
num_record = len(file)

statistic = file.groupby(['intersection_id','tollgate_id']).describe()
print(statistic)

pair_list = ['A-2','A-3','B-1','B-3','C-1','C-3']
threshold = {}
for each in pair_list:
    temp_list = each.split('-')
    if each not in threshold.keys():
        threshold[each] = {}
    threshold[each]['75%'] = statistic.loc[temp_list[0], int(temp_list[1]), '75%'][1]
    threshold[each]['50%'] = statistic.loc[temp_list[0], int(temp_list[1]), '50%'][1]


for i in range(num_record):
    print(i)
    temp_record = file.iloc[i]
    pair = str(temp_record.iloc[0]) + '-' + str(temp_record.iloc[1])
    if temp_record[-2] > threshold[pair]['75%']:
        file.iat[i,-1] = 2
    elif (temp_record[-2] <= threshold[pair]['75%']) & (temp_record[-2] > threshold[pair]['50%']):
        file.iat[i,-1] = 1
    else:
        file.iat[i,-1] = 0

file.to_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/trajectories(table 5)_training1.csv', columns=column, index=False)
del file

