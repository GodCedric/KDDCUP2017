# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
import datetime
import sklearn.neighbors
import sklearn.preprocessing

def crosswise_predict():

    # 训练数据
    # 读入横向数据
    crosswise_raw_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/result/training_20min_date_travel_time.csv')

    # 测试数据
    weather_raw_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/result/weather (table 7)_test1.csv')
    weather_raw_data = weather_raw_data.set_index(['date', 'hour'])

    # 输出
    fw = open('/home/godcedric/GitLocal/KDDCUP2017/submission_result/travel_time_submission_crosswise.csv', 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')

    # 横向预测
    weekday_dict = {1:'2016-10-24',
                    2:'2016-10-18',
                    3:'2016-10-19',
                    4:'2016-10-20',
                    5:'2016-10-21',
                    6:'2016-10-22',
                    7:'2016-10-23'}
    hour_dict = {8:6,
                 9:9,
                 17:15,
                 18:18}

    for i in range(len(crosswise_raw_data)):

        raw_sample = crosswise_raw_data.ix[i]

        intersection_id = raw_sample.intersection_id

        tollgate_id = raw_sample.tollgate_id

        temp_time_window = raw_sample[2].split(' ')

        time_window = '[' + weekday_dict[raw_sample.weekday] + ' ' + temp_time_window[0][1:] +',' + weekday_dict[raw_sample.weekday] + ' ' + temp_time_window[1]

        # 训练集
        temp_sample = raw_sample[4].split(';')
        for j in range(len(temp_sample)):
            per_sample = np.array(temp_sample[j].split(' '),dtype = np.float64)
            if j == 0:
                train_features = [list(per_sample[1:-1])]
                train_targets = [per_sample[-1]]
            else:
                train_features.append(list(per_sample[1:-1]))
                train_targets.append(per_sample[-1])

        train_features = np.array(train_features)
        train_targets = np.array(train_targets)

        # 归一化
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        train_features = min_max_scaler.fit_transform(train_features)


        # 测试集
        dateindex = weekday_dict[raw_sample.weekday]
        hourindex = hour_dict[datetime.datetime.strptime(temp_time_window[0][1:],"%H:%M:%S").hour]
        test_features = weather_raw_data.ix[dateindex].ix[hourindex][3:]

        # 训练
        n_neighbors = 1  # 选k个最近邻

        KNeighborsRegressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors)

        KNeighborsRegressor.fit(train_features, train_targets)

        # 预测
        test_features = np.array(test_features)
        avg_travel_time = KNeighborsRegressor.predict(test_features)

        # 输出
        out_line = ','.join(['"' + intersection_id + '"', '"' + str(tollgate_id) + '"',
                             '"' + time_window + '"',
                             '"' + str(avg_travel_time[0]) + '"']) + '\n'
        fw.writelines(out_line)

    fw.close()


def main():

    # 横向预测
    crosswise_predict()

if __name__ == '__main__':
    main()



