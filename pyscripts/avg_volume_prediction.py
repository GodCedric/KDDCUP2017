# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
计算各时间窗口的平均值，当作预测输出
"""

import math
from datetime import datetime,timedelta,time,date

file_suffix = '.csv'
path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/'  # set the data directory

def avg_volume_prediction(in_file):

    out_suffix = 'volume_red_timewindow_meanfill'
    in_file_name = in_file + file_suffix
    out_file_name = out_suffix + file_suffix

    # 步骤1：读文件
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    fr.close()

    # 步骤2：读取平均流量数据
    volumes = {}  # key: time window value: dictionary
    for i in range(len(vol_data)):
        each_pass = vol_data[i].replace('"', '').split(',')
        tollgate_id = each_pass[1]
        direction = each_pass[2]

        pass_time = each_pass[0]
        pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
        #print pass_time
        start_time_window = datetime(pass_time.year, pass_time.month, pass_time.day,
                                     pass_time.hour, time_window_minute, 0)

        if start_time_window not in volumes:
            volumes[start_time_window] = {}
        if tollgate_id not in volumes[start_time_window]:
            volumes[start_time_window][tollgate_id] = {}
        if direction not in volumes[start_time_window][tollgate_id]:
            volumes[start_time_window][tollgate_id][direction] = 1
        else:
            volumes[start_time_window][tollgate_id][direction] += 1

    # 步骤3：将平均流量按日期分开
    volumes2 = {}  #{收费站：{进出方向：{周一到周七：{时间段，[流量列表]}}}}
    time_reslut_sequence = ['08:00:00','08:20:00','08:40:00','09:00:00','09:20:00','09:40:00',
                            '17:00:00','17:20:00','17:40:00','18:00:00','18:20:00','18:40:00',]

    volumes2[1] = {0: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}},
                   1: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}}
    volumes2[2] = {0: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}}
    volumes2[3] = {0: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}},
                   1: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}}

    holiday = [date(2016,10,1),date(2016,10,2),date(2016,10,3),date(2016,10,4),date(2016,10,5),date(2016,10,6),date(2016,10,7)]

    for tm in volumes.keys():
        if tm.date not in holiday:
            if (tm.hour >= 8 and tm.hour<10) or (tm.hour>=17 and tm.hour<19):
                for toid in volumes[tm]:
                    for dir in volumes[tm][toid]:
                        vo = volumes[tm][toid][dir]
                        if tm.time() not in volumes2[int(toid)][int(dir)][tm.weekday()].keys():
                            volumes2[int(toid)][int(dir)][tm.weekday()][tm.time()] = [vo]
                        else:
                            volumes2[int(toid)][int(dir)][tm.weekday()][tm.time()].append(vo)


    # 步骤4：计算平均流量并输出
    cout_datetime = {}
    for i in range(6):
        cout_datetime[i+1] = date(2016,10,18) + timedelta(days = i)
    cout_datetime[0] = date(2016,10,24)

    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    for toid in volumes2.keys():
        for dir in volumes2[toid].keys():
            for da in volumes2[toid][dir].keys():
                for ti in volumes2[toid][dir][da].keys():
                    vo_set = volumes2[toid][dir][da][ti]
                    avg_vo = round(sum(vo_set) / float(len(vo_set)), 2)
                    date_cout = cout_datetime[da]
                    time_window_start = ti
                    cout_time_start = datetime(date_cout.year, date_cout.month, date_cout.day,
                                               time_window_start.hour, time_window_start.minute,
                                               time_window_start.second)
                    cout_time_end = cout_time_start + timedelta(minutes=20)
                    out_line = ','.join(['"' + str(toid) + '"',
                                         '"[' + str(cout_time_start) + ',' + str(cout_time_end) + ')"',
                                         '"' + str(dir) + '"',
                                         '"' + str(avg_vo) + '"',
                                         ]) + '\n'
                    fw.writelines(out_line)
    fw.close()

def main():

    in_file = 'volume(table 6)_training'
    avg_volume_prediction(in_file)

if __name__ == '__main__':
    main()