# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Calculate the average travel time for each 20-minute time window.

"""

# import necessary modules
import math
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from collections import defaultdict

file_suffix = '.csv'
#path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/'  # set the data directory
path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/testing_phase1/'

def avgTravelTime(in_file):

    out_suffix = '_link_info'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix
    out_file_path = '/home/godcedric/GitLocal/KDDCUP2017/特征工程2/时间窗数据/'
    out_file_name = out_file_path + out_file_name

    # Step 1: Load trajectories
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()
    print(traj_data[0])

    # Step 2: Create a dictionary to store travel time for each route per time window
    link_info = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')

        # 获取start_time
        trace_start_time = each_traj[3]
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)

        if start_time_window not in link_info.keys():
            link_info[start_time_window] = {}

        # 获取各link信息
        cur_link = each_traj[4]
        cur_link = cur_link.split(';')
        for x in cur_link:
            x = x.split('#')
            link_id = 'link_' + x[0]
            link_time = float(x[2])
            if link_id not in link_info[start_time_window].keys():
                link_info[start_time_window][link_id] = [link_time]
            else:
                link_info[start_time_window][link_id].append(link_time)


    # Step 3: Calculate average travel time for each route per time window
    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"time_window"', '"link_100"', '"link_101"', '"link_102"', '"link_103"', '"link_104"', '"link_105"',
                            '"link_106"', '"link_107"', '"link_108"', '"link_109"', '"link_110"', '"link_111"', '"link_112"',
                            '"link_113"', '"link_114"', '"link_115"', '"link_116"', '"link_117"', '"link_118"', '"link_119"',
                            '"link_120"', '"link_121"', '"link_122"', '"link_123"']) + '\n')
    for time_window_start in link_info.keys():
        link = list(link_info[time_window_start].keys())

        link_avg_time = defaultdict(int)
        for each_link in link:
            time_window_end = time_window_start + timedelta(minutes=20)
            tt_set = link_info[time_window_start][each_link]


            # 去除平均时间异常值
            def drop_outliers(data):
                data = sorted(data)
                if(len(data) <= 3):
                    return data
                meanall = round(sum(data) / float(len(data)), 2)
                if(data[-1]>meanall and data[-2]<meanall):
                    return data[:len(data)-1]
                else:
                    return data

            tt_set = drop_outliers(tt_set)

            avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
            link_avg_time[each_link] = avg_tt


        out_line = ','.join(['"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                             '"' + str(link_avg_time['link_100']) + '"', '"' + str(link_avg_time['link_101']) + '"',
                             '"' + str(link_avg_time['link_102']) + '"', '"' + str(link_avg_time['link_103']) + '"',
                             '"' + str(link_avg_time['link_104']) + '"', '"' + str(link_avg_time['link_105']) + '"',
                             '"' + str(link_avg_time['link_106']) + '"', '"' + str(link_avg_time['link_107']) + '"',
                             '"' + str(link_avg_time['link_108']) + '"', '"' + str(link_avg_time['link_109']) + '"',
                             '"' + str(link_avg_time['link_110']) + '"', '"' + str(link_avg_time['link_111']) + '"',
                             '"' + str(link_avg_time['link_112']) + '"', '"' + str(link_avg_time['link_113']) + '"',
                             '"' + str(link_avg_time['link_114']) + '"', '"' + str(link_avg_time['link_115']) + '"',
                             '"' + str(link_avg_time['link_116']) + '"', '"' + str(link_avg_time['link_117']) + '"',
                             '"' + str(link_avg_time['link_118']) + '"', '"' + str(link_avg_time['link_119']) + '"',
                             '"' + str(link_avg_time['link_120']) + '"', '"' + str(link_avg_time['link_121']) + '"',
                             '"' + str(link_avg_time['link_122']) + '"', '"' + str(link_avg_time['link_123']) + '"',
                             ]) + '\n'
        fw.writelines(out_line)



    fw.close()

def main():

    #in_file = 'trajectories(table 5)_training'
    in_file = 'trajectories(table 5)_test1'
    avgTravelTime(in_file)

if __name__ == '__main__':
    main()



