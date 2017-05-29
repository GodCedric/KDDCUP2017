# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

# 录入数据
test_travel_time = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/submission_sample_travelTime.csv')
test_volume = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/submission_sample_volume.csv')

test_travel_time['start_time'] = test_travel_time['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))
test_volume['start_time'] = test_volume['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))

test_travel_time['start_time'] = test_travel_time['start_time'].map(lambda x: x + timedelta(days=7))
test_volume['start_time'] = test_volume['start_time'].map(lambda x: x + timedelta(days=7))

test_travel_time['time_window'] = test_travel_time['start_time'].map(lambda x: '[' + str(x) + ',' +str(x+timedelta(minutes=20)) + ')')
test_volume['time_window'] = test_volume['start_time'].map(lambda x: '[' + str(x) + ',' +str(x+timedelta(minutes=20)) + ')')

test_travel_time = test_travel_time.sort_values(by=['intersection_id', 'tollgate_id', 'start_time'])
test_travel_time.index = np.arange(len(test_travel_time))
test_volume = test_volume.sort_values(by=['tollgate_id', 'direction', 'start_time'])
test_volume.index = np.arange(len(test_volume))

del test_travel_time['start_time']
del test_volume['start_time']

test_travel_time.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/原始数据/submission_sample_travelTime.csv', index=False)
test_volume.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/原始数据/submission_sample_volume.csv', index=False)