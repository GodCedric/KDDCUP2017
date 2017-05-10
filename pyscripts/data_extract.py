# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
提取数据，转换格式
"""

import numpy as np
import pandas as pd


# 提取travel_time数据
def extract_travel_time():

    in_file_path = '/home/godcedric/GitLocal/KDDCUP2017/result/training_20min_avg_travel_time.csv'

    raw_data = pd.read_csv(in_file_path)

    # 分析不同路径间









def main():

    extract_travel_time()


if __name__ == '__main__':
    main()
