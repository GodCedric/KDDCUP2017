# -*- coding: utf-8 -*-
#!/usr/bin/env python

import pandas as pd
from datetime import datetime,timedelta,date,time

travel_time_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/submission_sample_travelTime.csv')
volume_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/submission_sample_volume.csv')

travel_time_submission['start_time'] = travel_time_submission['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))
volume_submission['start_time'] = volume_submission['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))

travel_time_submission = travel_time_submission.sort_values(by = ['intersection_id', 'tollgate_id', 'start_time'])
del travel_time_submission['start_time']

volume_submission = volume_submission.sort_values(by = ['tollgate_id', 'direction', 'start_time'])
del volume_submission['start_time']

travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/travel_time_submission.csv', index=False)
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/volume_submission.csv', index=False)
