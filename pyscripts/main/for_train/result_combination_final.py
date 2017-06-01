# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd

mean_time_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.5/Mean/travel_time_submission.csv')

xgboost_time_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.5/xgboost_original/travel_time_submission.csv')

xgboost_time_result2 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.5/xgboost_new/travel_time_submission.csv')

RF_time_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.5/RF/travel_time_submission.csv')

last_result = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_submission/day1/travel_time_submission.csv')

xgboost_3 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_submission/submission_travelTime.csv')

travel_time_submission = mean_time_result[['intersection_id', 'tollgate_id', 'time_window']]

time_result_1 = mean_time_result['avg_travel_time']
time_result_2 = xgboost_time_result2['avg_travel_time']
time_result_3 = xgboost_3['avg_travel_time']
time_result_4 = RF_time_result['avg_travel_time']
#time_result_5 = last_result['avg_travel_time']

# 精细化融合
A2_result = 0.5*time_result_1[0:84] + 0.2*time_result_2[0:84] + 0.3*time_result_4[0:84]
A3_result = 0.4*time_result_1[84:168] + 0.2*time_result_2[84:168] + 0.4*time_result_4[84:168]
B1_result = 0.4*time_result_1[168:252] + 0.3*time_result_2[168:252] + 0.3*time_result_4[168:252]
B3_result = 0.5*time_result_1[252:336] + 0.5*time_result_3[252:336]
C1_result = (time_result_1[336:420] + time_result_2[336:420] + time_result_3[336:420] )/3
C3_result = 0.3*time_result_1[420:504] + 0.4*time_result_3[420:504] + 0.4*time_result_4[420:504]

time_predict = np.concatenate([A2_result,A3_result,B1_result,B3_result,C1_result,C3_result], axis=0)