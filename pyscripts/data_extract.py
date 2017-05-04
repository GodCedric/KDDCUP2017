# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
提取数据，转换格式
"""

import numpy as np
import pandas as pd

file_suffix = '.csv'
path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/'  # set the data directory

# 提取travel_time数据
def extract_travel_time(in_file):

    raw_data = pd.read_csv(in_file)
    print(raw_data.columns)
    print(raw_data.index)

    print('hello')
    print('hello')







def main():
    in_file = path + 'trajectories(table 5)_training' + file_suffix
    extract_travel_time(in_file)


if __name__ == '__main__':
    main()
