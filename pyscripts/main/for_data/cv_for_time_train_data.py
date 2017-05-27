# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

travel_time_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/5.0/travel_time/包含1个月的训练集和测试集/travel_time_train_data.csv')

week1 = ['2016-09-20', '2016-09-21', '2016-09-22', '2016-09-23', '2016-09-24', '2016-09-25', '2016-09-26']
week2 = ['2016-09-27', '2016-09-28', '2016-09-29', '2016-09-30', '2016-10-01', '2016-10-02', '2016-10-03']
week3 = ['2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07', '2016-10-08', '2016-10-09', '2016-10-10']
week4 = ['2016-10-11', '2016-10-12', '2016-10-13', '2016-10-14', '2016-10-15', '2016-10-16', '2016-10-17']

temp = travel_time_train_data.copy()
temp['index_in'] = temp['date'].map(lambda x: x in week4)
temp['index_out'] = temp['date'].map(lambda x: x not in week4)

val_data = travel_time_train_data[temp['index_in'].values]
train_data = travel_time_train_data[temp['index_out'].values]

train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/5.0/travel_time/1个月训练集分割4份/week4/train_data.csv', index=False)
val_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/5.0/travel_time/1个月训练集分割4份/week4/val_data.csv', index=False)