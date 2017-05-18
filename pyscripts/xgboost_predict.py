# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
XGBoost for prediction
Created on 2017/05/17
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import time

# 记录时间
start_tiem = time.time()

# 录入数据
travel_time_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/数据集生成/以最近值填充缺失值的训练集及测试集/travel_time_data_ffill.csv')
predict_raw_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/数据集生成/以最近值填充缺失值的训练集及测试集/test_travel_time_feature_ffill.csv')

#


