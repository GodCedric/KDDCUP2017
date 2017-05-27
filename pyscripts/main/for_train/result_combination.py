# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
结果融合
"""

import numpy as np
import pandas as pd

mean_time_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/mean1.5/2小时修正_系数/travel_time_submission.csv')
mean_volume_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/mean1.5/2小时修正_系数/volume_submission.csv')

xgboost_time_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/xgboost/非独热结果/travel_time_submission.csv')
xgboost_volume_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/xgboost/非独热结果/volume_submission.csv')

kNN_time_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1/kNN1.0/travel_time_submission.csv')
kNN_volume_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1/kNN1.0/volume_submission.csv')

RF_time_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/splitRF/travel_time_submission.csv')
RF_volume_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/splitRF/volume_submission.csv')


travel_time_submission = mean_time_result[['intersection_id', 'tollgate_id', 'time_window']]
volume_submission = mean_volume_result[['tollgate_id', 'time_window', 'direction']]

time_result_1 = mean_time_result['avg_travel_time']
time_result_2 = xgboost_time_result['avg_travel_time']
time_result_3 = kNN_time_result['avg_travel_time']
time_result_4 = RF_time_result['avg_travel_time']

time_result = 0.25*time_result_1 + 0.25*time_result_2 + 0.25*time_result_3 + 0.25*time_result_4

volume_result_1 = mean_volume_result['volume']
volume_result_2 = xgboost_volume_result['volume']
volume_result_3 = kNN_volume_result['volume']
volume_result_4 = RF_volume_result['volume']

volume_result = 0.25*volume_result_1 + 0.25*volume_result_2 + 0.25*volume_result_3 + 0.25*volume_result_4


travel_time_submission['avg_travel_time'] = time_result
volume_submission['volume'] = volume_result


travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/combination/travel_time_submission.csv', index=False)
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/combination/volume_submission.csv', index=False)