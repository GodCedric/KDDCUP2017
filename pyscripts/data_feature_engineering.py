# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
特征工程
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

def data_feature_engineering(travel_time_infile, volume_infile, test_travel_time_infile, test_volumn_infile):

    travel_time_data = pd.read_csv(travel_time_infile)
    volume_data = pd.read_csv(volume_infile)
    test_travel_time_data = pd.read_csv(test_travel_time_infile)
    test_volume_data = pd.read_csv(test_volumn_infile)

    # 删除一些列
    del travel_time_data['time_window']
    del travel_time_data['start_time']
    del travel_time_data['sea_pressure']
    del travel_time_data['date']
    del travel_time_data['time']
    del travel_time_data['holiday']
    del volume_data['time_window']
    del volume_data['etc']
    del volume_data['start_time']
    del volume_data['date']
    del volume_data['hour']
    del volume_data['sea_pressure']
    del volume_data['time']
    del test_travel_time_data['time_window']
    del test_travel_time_data['start_time']
    del test_travel_time_data['sea_pressure']
    del test_travel_time_data['date']
    del test_travel_time_data['time']
    del test_volume_data['time_window']
    del test_volume_data['start_time']
    del test_volume_data['date']
    del test_volume_data['sea_pressure']
    del test_volume_data['time']

    # 列排序
    time_columns = ['avg_travel_time', 'route', 'intersection_id', 'tollgate_id', 'weekday', 'timemap', 'pressure',
                    'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min',
                    'last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min']
    time_columns2 = ['route', 'intersection_id', 'tollgate_id', 'weekday', 'timemap', 'pressure', 'wind_direction',
                     'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min', 'last_40min',
                     'last_60min', 'last_80min', 'last_100min', 'last_120min']
    volume_columns = ['volume', 'pair', 'tollgate_id', 'direction', 'weekday', 'timemap', 'pressure', 'wind_direction',
                      'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min', 'last_40min',
                      'last_60min', 'last_80min', 'last_100min', 'last_120min']
    volume_columns2 = ['pair', 'tollgate_id', 'direction', 'weekday', 'timemap', 'pressure', 'wind_direction',
                       'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min', 'last_40min',
                       'last_60min', 'last_80min', 'last_100min', 'last_120min']
    travel_time_data = pd.DataFrame(travel_time_data, columns=time_columns)
    volume_data = pd.DataFrame(volume_data, columns=volume_columns)
    test_travel_time_data = pd.DataFrame(test_travel_time_data, columns=time_columns2)
    test_volume_data = pd.DataFrame(test_volume_data, columns=volume_columns2)

    # 风向（映射成东北风，东南风，西南风，西北风）
    def wind_direction_map(x):
        if 0 < x <= 90:
            return 1
        elif 90 < x <= 180:
            return 2
        elif 180 < x <= 270:
            return 3
        elif 270 < x <= 360:
            return 4

    travel_time_data['wind_direction2'] = travel_time_data['wind_direction'].map(wind_direction_map)
    volume_data['wind_direction2'] = volume_data['wind_direction'].map(wind_direction_map)
    test_travel_time_data['wind_direction2'] = test_travel_time_data['wind_direction'].map(wind_direction_map)
    test_volume_data['wind_direction2'] = test_volume_data['wind_direction'].map(wind_direction_map)

    # 风力，映射为几级风
    def wind_speed_map(x):
        if x == 0:
            return 0
        elif 0 < x <= 0.3:
            return 0
        elif 0.3 < x <= 1.6:
            return 1
        elif 1.6 < x <= 3.4:
            return 2
        elif 3.4 < x <= 5.5:
            return 3
        elif 5.5 < x <= 8.0:
            return 4
        elif 8.0 < x <= 10.8:
            return 5
        elif 10.8 < x <= 13.9:
            return 6
        elif 13.9 < x <= 17.2:
            return 7
        elif 17.2 < x <= 20.8:
            return 8
        else:
            return 9

    travel_time_data['wind_speed2'] = travel_time_data['wind_speed'].map(wind_direction_map)
    volume_data['wind_speed2'] = volume_data['wind_speed'].map(wind_direction_map)
    test_travel_time_data['wind_speed2'] = test_travel_time_data['wind_speed'].map(wind_direction_map)
    test_volume_data['wind_speed2'] = test_volume_data['wind_speed'].map(wind_direction_map)

    # 雨量等级
    def precipitation_map(x):
        if x == 0:
            return 0
        if 0 < x <= 10:
            return 1  # 小雨
        elif 10 < x <= 25:
            return 2  # 中雨
        elif 25 < x <= 50:
            return 3  # 大雨
        elif 50 < x <= 100:
            return 4  # 暴雨
        else:
            return 5

    travel_time_data['precipitation2'] = travel_time_data['precipitation'].map(wind_direction_map)
    volume_data['precipitation2'] = volume_data['precipitation'].map(wind_direction_map)
    test_travel_time_data['precipitation2'] = test_travel_time_data['precipitation'].map(wind_direction_map)
    test_volume_data['precipitation2'] = test_volume_data['precipitation'].map(wind_direction_map)

    # 增加人体舒适指数
    travel_time_data['SSD'] = (1.818 * travel_time_data['temperature'] + 18.18) * (0.88 + 0.002 * travel_time_data['rel_humidity']) + (travel_time_data['temperature'] - 32) / (45 - travel_time_data['temperature']) - 3.2 * travel_time_data['wind_speed'] + 18.2
    volume_data['SSD'] = (1.818 * volume_data['temperature'] + 18.18) * (0.88 + 0.002 * volume_data['rel_humidity']) + (volume_data['temperature'] - 32) / (45 -volume_data['temperature']) - 3.2 * volume_data['wind_speed'] + 18.2
    test_travel_time_data['SSD'] = (1.818 * test_travel_time_data['temperature'] + 18.18) * (0.88 + 0.002 * test_travel_time_data['rel_humidity']) + (test_travel_time_data['temperature'] - 32) / (45 - test_travel_time_data['temperature']) - 3.2 * test_travel_time_data['wind_speed'] + 18.2
    test_volume_data['SSD'] = (1.818 * test_volume_data['temperature'] + 18.18) * (0.88 + 0.002 * test_volume_data['rel_humidity']) + (test_volume_data['temperature'] - 32) / (45 - test_volume_data['temperature']) - 3.2 * test_volume_data['wind_speed'] + 18.2

    def SSD_map(x):
        if x > 86:
            return 4  # 很热
        if 80 < x <= 86:
            return 3  # 炎热
        elif 76 < x <= 80:
            return 2  # 便热
        elif 71 < x <= 76:
            return 1  # 偏暖
        elif 59 < x <= 71:
            return 0  # 舒适
        elif 51 < x <= 59:
            return -1  # 微凉
        elif 39 < x <= 51:
            return -2  # 清凉
        elif 26 < x <= 39:
            return -3  # 很冷
        else:
            return -4  # 寒冷

    travel_time_data['SSD_level'] = travel_time_data['SSD'].map(wind_direction_map)
    volume_data['SSD_level'] = volume_data['SSD'].map(wind_direction_map)
    test_travel_time_data['SSD_level'] = test_travel_time_data['SSD'].map(wind_direction_map)
    test_volume_data['SSD_level'] = test_volume_data['SSD'].map(wind_direction_map)

    # 写出数据
    travel_time_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/travel_time_train_data.csv')
    volume_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/volume_train_data.csv')
    test_travel_time_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/test_travel_time_data.csv')
    test_volume_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/test_volume_data.csv')



def main():
    travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/清洗过的数据/travel_time_clean_data_ffill.csv'
    volume_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/清洗过的数据/volume_clean_data_ffill.csv'
    test_travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/test_travel_time_feature_ffill.csv'
    test_volumn_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/test_volume_feature_ffill.csv'

    data_feature_engineering(travel_time_infile, volume_infile, test_travel_time_infile, test_volumn_infile)

if __name__ == '__main__':
    main()