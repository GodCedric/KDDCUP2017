# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Objective:
Calculate the average travel time for each 20-minute time window.

"""

# import necessary modules
import math
import pandas as pd
import numpy as np
from datetime import datetime,timedelta

file_suffix = '.csv'

def avgTravelTime(in_file):

    out_suffix = '_20min_avg_travel_time'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Step 1: Load trajectories
    fr = open(in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()

    # Step 2: Create a dictionary to store travel time for each route per time window
    travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
    for i in range(len(traj_data)):
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]

        route_id = intersection_id + '-' + tollgate_id
        if route_id not in travel_times.keys():
            travel_times[route_id] = {}

        # 计算开始时刻位于哪个时间窗口
        trace_start_time = each_traj[3]
        trace_start_time = datetime.strptime(trace_start_time, "%Y/%m/%d %H:%M")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        tt = float(each_traj[-2]) # travel time
        Is_jam = int(each_traj[-1])






        #将位于同一个开始时间窗口内所有样本的travel_time放入对应的子字典中
        if start_time_window not in travel_times[route_id].keys():
            travel_times[route_id][start_time_window] = {}
            travel_times[route_id][start_time_window]['tt'] = [tt]
            travel_times[route_id][start_time_window]['Is_jam'] = [Is_jam]
        else:
            travel_times[route_id][start_time_window]['tt'].append(tt)
            travel_times[route_id][start_time_window]['Is_jam'].append(Is_jam)






    # Step 3: Calculate average travel time for each route per time window
    fw = open(out_file_name, 'w')
    # 输出结果文件中每一列代表的意思
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    for route in travel_times.keys():
        route_time_windows = list(travel_times[route].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            time_window_end = time_window_start + timedelta(minutes=20)
            tt_set = travel_times[route][time_window_start]['tt']
            Is_jam = pd.DataFrame(np.array(travel_times[route][time_window_start]['Is_jam']),columns=['Is_jam'])
            Is_jam['Is_jam'] = Is_jam['Is_jam'].astype('category')
            stac_result = np.array(Is_jam.groupby('Is_jam').size())
            temp_max = stac_result.max()
            stac_result = list(stac_result)
            JamGrade = stac_result.index(temp_max)
            avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)#round函数进行四舍五入，精度是小数点后两位
            out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                                 '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                 '"' + str(avg_tt) + '"',
                                 '"' + str(JamGrade) + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()

def main():

    in_file = 'trajectories(table 5)_training1'
    avgTravelTime(in_file)

if __name__ == '__main__':
    main()



