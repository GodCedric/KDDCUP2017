# -*- coding: utf-8 -*-
# !/user/bin/env python

'''
objective:
1、将原始数据进行统计成所需的特征属性值，
2、并按照解题思路进行数据横纵向抽取，
3、进行特征空间构建
4、然后根据所得特征空间，对相应时间段内数据缺失部分采用特征近似的样本对应值来补全
    （取最近邻，或可采用一定数量相似样本的均值，或采用所有同类样本）
author：陆韬宇
'''

# 导入模块

import Data_transform as dt
import Data_format as df


def main():
    #定义原始数据集的输入路径与输出路径
    path_train_in = 'd:/KDD Cup 2017/dataSets/training/'
    path_train_out = 'd:/KDD Cup 2017/dataSets/training/DataProcessed/'  # set the data directory
    path_test_in = 'd:/KDD Cup 2017/dataSets/testing_phase1/'
    path_test_out = 'd:/KDD Cup 2017/dataSets/testing_phase1/DataProcessed/'

    #定义预处理得到数据的路劲
    path_train = 'd:/KDD Cup 2017/dataSets/training/DataProcessed/'
    path_test = 'd:/KDD Cup 2017/dataSets/testing_phase1/DataProcessed/'



    #训练数据转换
    #将原始数据转换成用于预测的平均消耗时间
    in_file_tt_o = 'trajectories(table 5)_training'
    dt.avgTravelTime(in_file_tt_o, path_train_in, path_train_out)
    #将原始数据转换成用于预测的车流量
    in_file_volume_o = 'volume(table 6)_training'
    dt.avgVolume(in_file_volume_o, path_train_in, path_train_out)


    #测试数据转换
    #将原始数据转换成用于预测的平均消耗时间
    in_file_tt_t = 'trajectories(table 5)_test1'
    dt.avgTravelTime(in_file_tt_t, path_test_in, path_test_out)
    #将原始数据转换成用于预测的车流量
    in_file_volume_t = 'volume(table 6)_test1'
    dt.avgVolume(in_file_volume_t, path_test_in, path_test_out)



    #训练数据抽取
    #将转化后的平均消耗时间数据抽取成预期数据格式
    in_file_att = 'training_20min_avg_travel_time'
    in_file_weather = 'weather (table 7)_training_update'
    df.dateTravelTime(in_file_att, in_file_weather, path_train)
    df.hourTravelTime(in_file_att, in_file_weather, path_train)
    #将转化后的车流量数据抽取成预期数据格式
    in_file_av = 'training_20min_avg_volume'
    df.dateVolume(in_file_av, in_file_weather, path_train)
    df.hourVolume(in_file_av, in_file_weather, path_train)

    #将二次转换后的数据按样本格式整理好
    in_file_att = 'training_20min_hour_travel_time'
    df.hourTravelTimeSample(in_file_att, in_file_weather, path_train)
    in_file_av = 'training_20min_hour_volume'
    df.hourVolumeSample(in_file_av, in_file_weather, path_train)


    #测试数据抽取
    in_file_at_t = 'test1_20min_avg_travel_time'
    in_file_weather_t = 'weather (table 7)_test1'
    df.hourTravelTime(in_file_at_t, in_file_weather_t, path_test)
    in_file_at_t = 'test1_20min_hour_travel_time'
    df.hourTravelTimeSample(in_file_at_t, in_file_weather_t, path_test)
    df.TestWeatherDataTravelTime('test1_20min_hour_travel_time_sample', path_test)

    in_file_av_t = 'test1_20min_avg_volume'
    df.hourVolume(in_file_av_t, in_file_weather_t, path_test)
    in_file_av_t = 'test1_20min_hour_volume'
    df.hourVolumeSample(in_file_av_t, in_file_weather_t, path_test)
    df.TestWeatherDataVolume('test1_20min_hour_volume_sample', path_test)



if __name__ == '__main__':
    main()