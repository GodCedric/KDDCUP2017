# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

# 录入数据
trajectories_phase1 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/trajectories(table 5)_training.csv')
volume_phase1 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/volume(table 6)_training.csv')

trajectories_phase2 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSet_phase2/trajectories(table_5)_training2.csv')
volume_phase2 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSet_phase2/volume(table 6)_training2.csv')

trajectories_test2 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSet_phase2/trajectories(table 5)_test2.csv')
volume_test2 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSet_phase2/volume(table 6)_test2.csv')

# 合并
trajectories_train = pd.concat([trajectories_phase1, trajectories_phase2], axis=0)
volume_train = pd.concat([volume_phase1, volume_phase2], axis=0)

trajectories_test = trajectories_test2
volume_test = volume_test2

trajectories_train.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/trajectories_train.csv', index=False)
volume_train.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/volume_train.csv', index=False)

trajectories_test.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/trajectories_test.csv', index=False)
volume_test.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/volume_test.csv', index=False)