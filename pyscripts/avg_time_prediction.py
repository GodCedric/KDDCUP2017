#coding=utf-8
#!/usr/bin/env python

"""
计算各时间窗口的平均值，当作预测输出
"""

import math
from datetime import datetime,timedelta,time,date

file_suffix = '.csv'
path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/'  # set the data directory

def avg_time_prediction(in_file):

    out_suffix = 'avg_time_submission'
    in_file_name = in_file + file_suffix
    out_file_name = out_suffix + file_suffix

    # 步骤1：读文件
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()

    # 步骤2：读取平均时间数据，按照星期几分割开
    travel_times = {}  # {A2到C3：{周一到周七:{时间段，[平均时间列表]}}}
    for i in range(len(traj_data)):
        #   获取路线段
        each_traj = traj_data[i].replace('"', '').split(',')
        intersection_id = each_traj[0]
        tollgate_id = each_traj[1]

        route_id = intersection_id + '-' + tollgate_id
        if route_id not in travel_times.keys():
            travel_times[route_id] = {}

        #   获取时间段
        trace_start_time = each_traj[3]
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_datetime_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        start_time_window = start_datetime_window.time()


        weekday = start_datetime_window.weekday()
        if weekday not in travel_times[route_id].keys():
            travel_times[route_id][weekday] = {}

        #   填平均时间
        tt = float(each_traj[-1]) # travel time
        if start_time_window not in travel_times[route_id][weekday].keys():
            travel_times[route_id][weekday][start_time_window] = [tt]
        else:
            travel_times[route_id][weekday][start_time_window].append(tt)


    # 步骤3：计算各时间窗口的平均时间并输出
    cout_datetime = {}
    for i in range(6):
        cout_datetime[i+1] = date(2016,10,18) + timedelta(days = i)
    cout_datetime[0] = date(2016,10,24)

    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    for route in travel_times.keys():
        for wkday in travel_times[route].keys():
            route_time_windows = list(travel_times[route][wkday].keys())
            route_time_windows.sort()
            result_time_windows = [rt for rt in route_time_windows if (rt.hour>=8 and rt.hour<10) or (rt.hour>=17 and rt.hour<19)]
            for time_window_start in result_time_windows:
                tt_set = travel_times[route][wkday][time_window_start]
                avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
                date_cout = cout_datetime[wkday]
                cout_time_start = datetime(date_cout.year, date_cout.month, date_cout.day,
                                           time_window_start.hour, time_window_start.minute, time_window_start.second)
                cout_time_end = cout_time_start + timedelta(minutes=20)
                out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                                    '"[' + str(cout_time_start) + ',' + str(cout_time_end) + ')"',
                                    '"' + str(avg_tt) + '"']) + '\n'
                fw.writelines(out_line)
    fw.close()

def main():

    in_file = 'trajectories(table 5)_training'
    avg_time_prediction(in_file)

if __name__ == '__main__':
    main()