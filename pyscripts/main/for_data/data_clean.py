# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
数据清洗
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

def data_clean(travel_time_infile, volume_infile):

    # 原始数据
    travel_time_data = pd.read_csv(travel_time_infile)
    volume_data = pd.read_csv(volume_infile)

    #####-----平均时间-----#####

    # 异常样本删除（以平均时间3标准差原则删除异常样本）
    """
    def ff(df, column='avg_travel_time'):
        travel_time = df['avg_travel_time']
        mean_value = travel_time.mean()
        std_value = travel_time.std()
        left = mean_value - 3*std_value
        right = mean_value + 3*std_value
        travel_time[travel_time < left] = np.nan
        travel_time[travel_time > right] = np.nan
        df = df.dropna()
        return df
    travel_time_data = travel_time_data.groupby('route').apply(ff)
    """
    # 异常样本删除
    def ff(df):
        df = df.sort(['avg_travel_time'], ascending=False)
        num_sample = len(df)
        num_delete = round(0.005 * num_sample)
        return df.iloc[num_delete:]

    travel_time_data = travel_time_data.groupby('route').apply(ff)

    # 风向异常值处理，以最近的风向值代替
    wind_direction = travel_time_data['wind_direction']
    wind_direction[wind_direction > 360] = np.nan
    wind_direction.fillna(method='ffill', inplace=True)

    # last缺失值处理
    """
    ### 以平均值填充缺失值
    fill_mean = lambda g:g.fillna(g.mean())

    travel_time_data['last_20min'] = travel_time_data.groupby(['route','time'])['last_20min'].apply(fill_mean)
    travel_time_data['last_40min'] = travel_time_data.groupby(['route','time'])['last_40min'].apply(fill_mean)
    travel_time_data['last_60min'] = travel_time_data.groupby(['route','time'])['last_60min'].apply(fill_mean)
    travel_time_data['last_80min'] = travel_time_data.groupby(['route','time'])['last_80min'].apply(fill_mean)
    travel_time_data['last_100min'] = travel_time_data.groupby(['route','time'])['last_100min'].apply(fill_mean)
    travel_time_data['last_120min'] = travel_time_data.groupby(['route','time'])['last_120min'].apply(fill_mean)
    """

    ### 最近值填充
    ffill = lambda g:g.fillna(method='ffill')
    bfill = lambda g:g.fillna(method='bfill')

    travel_time_data['last_20min'] = travel_time_data.groupby(['route'])['last_20min'].apply(ffill)
    travel_time_data['last_20min'] = travel_time_data.groupby(['route'])['last_20min'].apply(bfill)
    travel_time_data['last_40min'] = travel_time_data.groupby(['route'])['last_40min'].apply(ffill)
    travel_time_data['last_40min'] = travel_time_data.groupby(['route'])['last_40min'].apply(bfill)
    travel_time_data['last_60min'] = travel_time_data.groupby(['route'])['last_60min'].apply(ffill)
    travel_time_data['last_60min'] = travel_time_data.groupby(['route'])['last_60min'].apply(bfill)
    travel_time_data['last_80min'] = travel_time_data.groupby(['route'])['last_80min'].apply(ffill)
    travel_time_data['last_80min'] = travel_time_data.groupby(['route'])['last_80min'].apply(bfill)
    travel_time_data['last_100min'] = travel_time_data.groupby(['route'])['last_100min'].apply(ffill)
    travel_time_data['last_100min'] = travel_time_data.groupby(['route'])['last_100min'].apply(bfill)
    travel_time_data['last_120min'] = travel_time_data.groupby(['route'])['last_120min'].apply(ffill)
    travel_time_data['last_120min'] = travel_time_data.groupby(['route'])['last_120min'].apply(bfill)

    """
    # 是否删除节假日样本
    dropindex = travel_time_data[travel_time_data['holiday'] == 1].index
    travel_time_data = travel_time_data.drop(dropindex, axis=0)
    del travel_time_data['holiday']
    """

    # 写出数据
    travel_time_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/清洗数据/travel_time_clean_data_ffill.csv', index=False)


    #####-----流量-----#####


    # 删除节假日
    dropindex = volume_data[(volume_data['holiday'] == 1) & (volume_data['pair'] != '3-0')].index
    volume_data = volume_data.drop(dropindex, axis=0)
    dropindex = volume_data[(volume_data['pair'] == '1-0') & (volume_data['date'] == '2016-09-30')].index
    volume_data = volume_data.drop(dropindex, axis=0)
    del volume_data['holiday']


    # 风向异常值处理，以最近的风向值代替
    wind_direction = volume_data['wind_direction']
    wind_direction[wind_direction > 360] = np.nan
    wind_direction.fillna(method='ffill', inplace=True)

    # last值缺失处理
    """
    ### 以平均值填充缺失值
    volume_data['last_20min'] = volume_data.groupby(['pair', 'time'])['last_20min'].apply(fill_mean)
    volume_data['last_40min'] = volume_data.groupby(['pair', 'time'])['last_40min'].apply(fill_mean)
    volume_data['last_60min'] = volume_data.groupby(['pair', 'time'])['last_60min'].apply(fill_mean)
    volume_data['last_80min'] = volume_data.groupby(['pair', 'time'])['last_80min'].apply(fill_mean)
    volume_data['last_100min'] = volume_data.groupby(['pair', 'time'])['last_100min'].apply(fill_mean)
    volume_data['last_120min'] = volume_data.groupby(['pair', 'time'])['last_120min'].apply(fill_mean)
    """

    # 最近值填充
    volume_data['last_20min'] = volume_data.groupby(['pair'])['last_20min'].apply(ffill)
    volume_data['last_20min'] = volume_data.groupby(['pair'])['last_20min'].apply(bfill)
    volume_data['last_40min'] = volume_data.groupby(['pair'])['last_40min'].apply(ffill)
    volume_data['last_40min'] = volume_data.groupby(['pair'])['last_40min'].apply(bfill)
    volume_data['last_60min'] = volume_data.groupby(['pair'])['last_60min'].apply(ffill)
    volume_data['last_60min'] = volume_data.groupby(['pair'])['last_60min'].apply(bfill)
    volume_data['last_80min'] = volume_data.groupby(['pair'])['last_80min'].apply(ffill)
    volume_data['last_80min'] = volume_data.groupby(['pair'])['last_80min'].apply(bfill)
    volume_data['last_100min'] = volume_data.groupby(['pair'])['last_100min'].apply(ffill)
    volume_data['last_100min'] = volume_data.groupby(['pair'])['last_100min'].apply(bfill)
    volume_data['last_120min'] = volume_data.groupby(['pair'])['last_120min'].apply(ffill)
    volume_data['last_120min'] = volume_data.groupby(['pair'])['last_120min'].apply(bfill)

    # 写出数据
    volume_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/清洗数据/volume_clean_data_ffill.csv', index=False)

def main():

    #travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/travel_time_raw_data.csv'
    #volume_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/volume_raw_data.csv'

    travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/final_data/提取数据/travel_time_raw_data.csv'
    volume_infile = '/home/godcedric/GitLocal/KDDCUP2017/final_data/提取数据/volume_raw_data.csv'

    data_clean(travel_time_infile, volume_infile)


if __name__ == '__main__':
    main()