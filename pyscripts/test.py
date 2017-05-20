# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

travel_time_data = pd.read_csv('/home/godcedric/Jupyter_Notebook/数据预处理/travel_time_raw_data.csv')
volume_data = pd.read_csv('/home/godcedric/Jupyter_Notebook/数据预处理/volume_raw_data.csv')

def ff(df, column='avg_travel_time'):
    travel_time = df['avg_travel_time']
    mean_value = travel_time.mean()
    std_value = travel_time.std()
    left = mean_value - 3*std_value
    right = mean_value + 3*std_value
    travel_time[travel_time < left] = np.nan
    travel_time[travel_time > right] = np.nan
    df.dropna()

travel_time_data.groupby('route').apply(ff)