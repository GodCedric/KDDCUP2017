# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
特征工程
"""
import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time
from sklearn import preprocessing

def data_feature_engineering(travel_time_infile, volume_infile, test_travel_time_infile, test_volumn_infile):

    travel_time_data = pd.read_csv(travel_time_infile)
    volume_data = pd.read_csv(volume_infile)
    test_travel_time_data = pd.read_csv(test_travel_time_infile)
    test_volume_data = pd.read_csv(test_volumn_infile)


    """
    # 平均时间数据取近一个月的
    start_date = date(2016, 9, 20)
    travel_time_data['date2'] = pd.to_datetime(travel_time_data['date'], format='%Y-%m-%d')
    travel_time_data = travel_time_data[travel_time_data['date2'] >= start_date]
    del travel_time_data['date2']

    # 流量数据取10月7号之后的
    start_date = date(2016, 10, 8)
    volume_data['date2'] = pd.to_datetime(volume_data['date'], format='%Y-%m-%d')
    volume_data = volume_data[volume_data['date2'] >= start_date]
    del volume_data['date2']
    """


    # 删除一些列
    #del travel_time_data['time_window']
    #del travel_time_data['start_time']
    del travel_time_data['sea_pressure']
    #del travel_time_data['date']
    #del travel_time_data['time']
    del travel_time_data['holiday']
    #del volume_data['time_window']
    #del volume_data['etc']
    #del volume_data['start_time']
    #del volume_data['date']
    #del volume_data['hour']
    del volume_data['sea_pressure']
    #del volume_data['time']
    #del test_travel_time_data['time_window']
    #del test_travel_time_data['start_time']
    del test_travel_time_data['sea_pressure']
    #del test_travel_time_data['date']
    #del test_travel_time_data['time']
    #del test_volume_data['time_window']
    #del test_volume_data['start_time']
    #del test_volume_data['date']
    del test_volume_data['sea_pressure']
    #del test_volume_data['time']

    # 列排序
    time_columns = ['avg_travel_time', 'route', 'intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time', 'hour', 'minute', 'weekday', 'timemap', 'pressure',
                    'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min',
                    'last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min']
    time_columns2 = ['route', 'intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time', 'hour', 'minute', 'weekday', 'timemap', 'pressure', 'wind_direction',
                     'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min', 'last_40min',
                     'last_60min', 'last_80min', 'last_100min', 'last_120min']
    volume_columns = ['volume', 'pair', 'tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time', 'hour', 'minute', 'weekday', 'timemap', 'pressure', 'wind_direction',
                      'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min', 'last_40min',
                      'last_60min', 'last_80min', 'last_100min', 'last_120min']
    volume_columns2 = ['pair', 'tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time', 'hour', 'minute', 'weekday', 'timemap', 'pressure', 'wind_direction',
                       'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'last_20min', 'last_40min',
                       'last_60min', 'last_80min', 'last_100min', 'last_120min']
    travel_time_data = pd.DataFrame(travel_time_data, columns=time_columns)
    volume_data = pd.DataFrame(volume_data, columns=volume_columns)
    test_travel_time_data = pd.DataFrame(test_travel_time_data, columns=time_columns2)
    test_volume_data = pd.DataFrame(test_volume_data, columns=volume_columns2)

    # 风向（映射成东风，西风，南风，北风，东北风，东南风，西南风，西北风）
    def wind_direction_map(x):
        if 22.5 < x <= 67.5:
            return 1
        elif 67.5 < x <= 112.5:
            return 2
        elif 112.5 < x <= 157.5:
            return 3
        elif 157.5 < x <= 202.5:
            return 4
        elif 202.5 < x <= 247.5:
            return 5
        elif 247.5 < x <= 292.5:
            return 6
        elif 292.5 < x <= 337.5:
            return 7
        else:
            return 8

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

    travel_time_data['wind_speed2'] = travel_time_data['wind_speed'].map(wind_speed_map)
    volume_data['wind_speed2'] = volume_data['wind_speed'].map(wind_speed_map)
    test_travel_time_data['wind_speed2'] = test_travel_time_data['wind_speed'].map(wind_speed_map)
    test_volume_data['wind_speed2'] = test_volume_data['wind_speed'].map(wind_speed_map)

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

    travel_time_data['precipitation2'] = travel_time_data['precipitation'].map(precipitation_map)
    volume_data['precipitation2'] = volume_data['precipitation'].map(precipitation_map)
    test_travel_time_data['precipitation2'] = test_travel_time_data['precipitation'].map(precipitation_map)
    test_volume_data['precipitation2'] = test_volume_data['precipitation'].map(precipitation_map)

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

    travel_time_data['SSD_level'] = travel_time_data['SSD'].map(SSD_map)
    volume_data['SSD_level'] = volume_data['SSD'].map(SSD_map)
    test_travel_time_data['SSD_level'] = test_travel_time_data['SSD'].map(SSD_map)
    test_volume_data['SSD_level'] = test_volume_data['SSD'].map(SSD_map)

    # 星期几映射成workday（工作日为1，周末为2，节假日为3）
    holiday = ['2016-09-15', '2016-09-16', '2016-09-17', '2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04',
               '2016-10-05', '2016-10-06', '2016-10-07']

    def ff(x):
        if x in holiday:
            return 3
        else:
            format_time = datetime.strptime(x, '%Y-%m-%d')
            weekday = format_time.weekday()
            if (weekday == 5 or weekday == 6):
                return 2
            else:
                return 1

    travel_time_data['is_workday'] = travel_time_data['date'].map(ff)
    volume_data['is_workday'] = volume_data['date'].map(ff)
    test_travel_time_data['is_workday'] = test_travel_time_data['date'].map(ff)
    test_volume_data['is_workday'] = test_volume_data['date'].map(ff)

    # pressure取整
    """
    travel_time_data['pressure'] = travel_time_data['pressure'].map(lambda x: round(x))
    volume_data['pressure'] = volume_data['pressure'].map(lambda x: round(x))
    test_travel_time_data['pressure'] = test_travel_time_data['pressure'].map(lambda x: round(x))
    test_volume_data['pressure'] = test_volume_data['pressure'].map(lambda x: round(x))
    """

    # 增加datemap特征
    travel_time_data['datetemp'] = travel_time_data['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    volume_data['datetemp'] = volume_data['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    test_travel_time_data['datetemp'] = test_travel_time_data['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    test_volume_data['datetemp'] = test_volume_data['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    start_time1 = datetime(2016, 7, 18, 0, 0, 0)
    start_time2 = datetime(2016, 9, 18, 0, 0, 0)
    start_time3 = datetime(2016, 10, 17, 0, 0, 0)
    travel_time_data['datemap'] = travel_time_data['datetemp'].map(lambda x: int((x - start_time1).days))
    volume_data['datemap'] = volume_data['datetemp'].map(lambda x: int((x - start_time2).days))
    test_travel_time_data['datemap'] = test_travel_time_data['datetemp'].map(lambda x: int((x - start_time3).days))
    test_volume_data['datemap'] = test_volume_data['datetemp'].map(lambda x: int((x - start_time3).days))
    del travel_time_data['datetemp']
    del volume_data['datetemp']
    del test_travel_time_data['datetemp']
    del test_volume_data['datetemp']

    # 增加路径信息
    """
    road = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/pyscripts/analyze/road.csv')
    travel_time_data = pd.merge(travel_time_data, road, on=['intersection_id', 'tollgate_id'], how='left')
    test_travel_time_data = pd.merge(test_travel_time_data, road, on=['intersection_id', 'tollgate_id'], how='left')
    """

    # 离散特征独热编码
    # 平均时间
    route1 = travel_time_data.route
    route2 = test_travel_time_data.route
    route_all = pd.concat([route1, route2], axis=0)
    route_onehot = pd.get_dummies(route_all)
    travel_time_data = pd.concat([travel_time_data, route_onehot[:len(route1)]], axis=1)
    test_travel_time_data = pd.concat([test_travel_time_data, route_onehot[len(route1):]], axis=1)

    hour1 = travel_time_data.hour
    hour2 = test_travel_time_data.hour
    hour_all = pd.concat([hour1, hour2], axis=0)
    hour_onehot = pd.get_dummies(hour_all, prefix='hour_')
    travel_time_data = pd.concat([travel_time_data, hour_onehot[:len(hour1)]], axis=1)
    test_travel_time_data = pd.concat([test_travel_time_data, hour_onehot[len(hour1):]], axis=1)

    minute1 = travel_time_data.minute
    minute2 = test_travel_time_data.minute
    minute_all = pd.concat([minute1, minute2], axis=0)
    minute_onehot = pd.get_dummies(minute_all, prefix='minute_')
    travel_time_data = pd.concat([travel_time_data, minute_onehot[:len(minute1)]], axis=1)
    test_travel_time_data = pd.concat([test_travel_time_data, minute_onehot[len(minute1):]], axis=1)

    weekday1 = travel_time_data.weekday
    weekday2 = test_travel_time_data.weekday
    weekday_all = pd.concat([weekday1, weekday2], axis=0)
    weekday_onehot = pd.get_dummies(weekday_all, prefix='weekday_')
    travel_time_data = pd.concat([travel_time_data, weekday_onehot[:len(weekday1)]], axis=1)
    test_travel_time_data = pd.concat([test_travel_time_data, weekday_onehot[len(weekday1):]], axis=1)

    workday1 = travel_time_data.is_workday
    workday2 = test_travel_time_data.is_workday
    workday_all = pd.concat([workday1, workday2], axis=0)
    workday_onehot = pd.get_dummies(workday_all, prefix='workday_')
    travel_time_data = pd.concat([travel_time_data, workday_onehot[:len(workday1)]], axis=1)
    test_travel_time_data = pd.concat([test_travel_time_data, workday_onehot[len(workday1):]], axis=1)

    # 流量
    pair1 = volume_data.pair
    pair2 = test_volume_data.pair
    pair_all = pd.concat([pair1, pair2], axis=0)
    pair_onehot = pd.get_dummies(pair_all)
    volume_data = pd.concat([volume_data, pair_onehot[:len(pair1)]], axis=1)
    test_volume_data = pd.concat([test_volume_data, pair_onehot[len(pair1):]], axis=1)

    hour1 = volume_data.hour
    hour2 = test_volume_data.hour
    hour_all = pd.concat([hour1, hour2], axis=0)
    hour_onehot = pd.get_dummies(hour_all, prefix='hour_')
    volume_data = pd.concat([volume_data, hour_onehot[:len(hour1)]], axis=1)
    test_volume_data = pd.concat([test_volume_data, hour_onehot[len(hour1):]], axis=1)

    minute1 = volume_data.minute
    minute2 = test_volume_data.minute
    minute_all = pd.concat([minute1, minute2], axis=0)
    minute_onehot = pd.get_dummies(minute_all, prefix='minute_')
    volume_data = pd.concat([volume_data, minute_onehot[:len(minute1)]], axis=1)
    test_volume_data = pd.concat([test_volume_data, minute_onehot[len(minute1):]], axis=1)

    weekday1 = volume_data.weekday
    weekday2 = test_volume_data.weekday
    weekday_all = pd.concat([weekday1, weekday2], axis=0)
    weekday_onehot = pd.get_dummies(weekday_all, prefix='weekday_')
    volume_data = pd.concat([volume_data, weekday_onehot[:len(weekday1)]], axis=1)
    test_volume_data = pd.concat([test_volume_data, weekday_onehot[len(weekday1):]], axis=1)

    workday1 = volume_data.is_workday
    workday2 = test_volume_data.is_workday
    workday_all = pd.concat([workday1, workday2], axis=0)
    workday_onehot = pd.get_dummies(workday_all, prefix='workday_')
    volume_data = pd.concat([volume_data, workday_onehot[:len(workday1)]], axis=1)
    test_volume_data = pd.concat([test_volume_data, workday_onehot[len(workday1):]], axis=1)


    # 排好序
    travel_time_data = travel_time_data.sort_values(by = ['intersection_id', 'tollgate_id', 'start_time'])
    test_travel_time_data = test_travel_time_data.sort_values(by=['intersection_id', 'tollgate_id', 'start_time'])
    volume_data = volume_data.sort_values(by = ['tollgate_id', 'direction', 'start_time'])
    test_volume_data = test_volume_data.sort_values(by=['tollgate_id', 'direction', 'start_time'])

    # 写出数据
    travel_time_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/5.5/travel_time_train_data.csv', index=False)
    volume_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/5.5/volume_train_data.csv', index=False)
    test_travel_time_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/5.5/test_travel_time_data.csv', index=False)
    test_volume_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/5.5/test_volume_data.csv', index=False)



def main():
    #travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/清洗过的数据/travel_time_clean_data_ffill.csv'
    #volume_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/清洗过的数据/volume_clean_data_ffill.csv'
    #test_travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/test_travel_time_feature_ffill.csv'
    #test_volumn_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/test_volume_feature_ffill.csv'

    travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/final_data/清洗数据/travel_time_clean_data_ffill.csv'
    volume_infile = '/home/godcedric/GitLocal/KDDCUP2017/final_data/清洗数据/volume_clean_data_ffill.csv'
    test_travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/final_data/提取数据/test_travel_time_feature_ffill.csv'
    test_volumn_infile = '/home/godcedric/GitLocal/KDDCUP2017/final_data/提取数据/test_volume_feature_ffill.csv'


    data_feature_engineering(travel_time_infile, volume_infile, test_travel_time_infile, test_volumn_infile)

if __name__ == '__main__':
    main()