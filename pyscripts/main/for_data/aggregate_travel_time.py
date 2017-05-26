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

file_suffix = '.csv'
#path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/'  # set the data directory
path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/testing_phase1/'

def avgTravelTime(in_file):

    out_suffix = '_20min_avg_travel_time'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix
    out_file_path = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/初始形成时间窗的数据集/'
    out_file_name = out_file_path + out_file_name

    # Step 1: Load trajectories
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()
    print(traj_data[0])

    # Step 2: Create a dictionary to store travel time for each route per time window
    travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]

        route_id = intersection_id + '-' + tollgate_id
        if route_id not in travel_times.keys():
            travel_times[route_id] = {}

        trace_start_time = each_traj[3]
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        tt = float(each_traj[-1]) # travel time

        if start_time_window not in travel_times[route_id].keys():
            travel_times[route_id][start_time_window] = [tt]
        else:
            travel_times[route_id][start_time_window].append(tt)

    # Step 3: Calculate average travel time for each route per time window
    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    for route in travel_times.keys():
        route_time_windows = list(travel_times[route].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            time_window_end = time_window_start + timedelta(minutes=20)
            tt_set = travel_times[route][time_window_start]

            """
            # 去除平均时间异常值
            tt_set = [x for x in tt_set if x>0 and x<500]

            if len(tt_set)>0:
                avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
                time_set = ';'.join([str(e) for e in tt_set])
                out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                                     '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                     '"' + str(avg_tt) + '"']) + '\n'
                fw.writelines(out_line)
            """

            # 去除平均时间异常值
            avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
            time_set = ';'.join([str(e) for e in tt_set])
            out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                                '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                '"' + str(avg_tt) + '"']) + '\n'
            fw.writelines(out_line)

    fw.close()

def main():

    #in_file = 'trajectories(table 5)_training'
    in_file = 'trajectories(table 5)_test1'
    avgTravelTime(in_file)

if __name__ == '__main__':
    main()



