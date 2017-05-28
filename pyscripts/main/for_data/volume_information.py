# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
Calculate volume for each 20-minute time window.
"""
import math
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

file_suffix = '.csv'
#path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/'
path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/testing_phase1/'  # set the data directory

def avgVolume(in_file):
    out_suffix = '_volume_info'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix
    out_file_path = '/home/godcedric/GitLocal/KDDCUP2017/特征工程2/时间窗数据/'
    out_file_name = out_file_path + out_file_name

    # Step 1: Load volume data
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    fr.close()

    # Step 2: Create a dictionary to caculate and store volume per time window
    volumes_info = {}
    for i in range(len(vol_data)):
        each_pass = vol_data[i].replace('"', '').split(',')
        tollgate_id = each_pass[1]
        direction = each_pass[2]

        pass_time = each_pass[0]
        pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
        # print pass_time
        start_time_window = datetime(pass_time.year, pass_time.month, pass_time.day,
                                     pass_time.hour, time_window_minute, 0)


        # 加各v_model的数量
        cur_model = 'model' + '_' +each_pass[3]
        has_etc = int(each_pass[4])

        if tollgate_id not in volumes_info:
            volumes_info[tollgate_id] = {}
        if direction not in volumes_info[tollgate_id]:
            volumes_info[tollgate_id][direction] = {}
        if start_time_window not in volumes_info[tollgate_id][direction]:
            volumes_info[tollgate_id][direction][start_time_window] = defaultdict(int)

        # 录入信息
        volumes_info[tollgate_id][direction][start_time_window]['volume'] += 1
        volumes_info[tollgate_id][direction][start_time_window]['etc'] += has_etc
        volumes_info[tollgate_id][direction][start_time_window][cur_model] += 1
        if (each_pass[5] != '\n'):
            cur_type = 'type' + '_' + str(int(each_pass[5]))
            volumes_info[tollgate_id][direction][start_time_window][cur_type] += 1


        # Step 3: format output for tollgate and direction per time window
    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"', '"etc"', '"model_1"', '"model_2"', '"model_3"', '"model_4"', '"model_5"', '"model_6"', '"model_7"', '"type_0"', '"type_1"']) + '\n')
    for tollgate_id in volumes_info:
        for direction in volumes_info[tollgate_id]:
            time_window = list(volumes_info[tollgate_id][direction].keys())
            time_window.sort()
            for time_window_start in time_window:
                time_window_end = time_window_start + timedelta(minutes=20)
                out_line = ','.join(['"' + str(tollgate_id) + '"',
                                     '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                     '"' + str(direction) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['volume']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['etc']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['model_1']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['model_2']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['model_3']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['model_4']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['model_5']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['model_6']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['model_7']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['type_0']) + '"',
                                     '"' + str(volumes_info[tollgate_id][direction][time_window_start]['type_1']) + '"',
                                     ]) + '\n'
                fw.writelines(out_line)
    fw.close()


def main():
    #in_file = 'volume(table 6)_training'
    in_file = 'volume(table 6)_test1'
    avgVolume(in_file)


if __name__ == '__main__':
    main()



