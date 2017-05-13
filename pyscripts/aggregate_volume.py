# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Calculate volume for each 20-minute time window.
"""
import math
from datetime import datetime,timedelta

file_suffix = '.csv'
path = '/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/testing_phase1/'  # set the data directory

def avgVolume(in_file):

    out_suffix = '_20min_avg_volume'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Step 1: Load volume data
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    fr.close()

    # Step 2: Create a dictionary to caculate and store volume per time window
    volumes = {}  # key: time window value: dictionary
    etc = {}
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

        if tollgate_id not in volumes:
            volumes[tollgate_id] = {}
        if direction not in volumes[tollgate_id]:
            volumes[tollgate_id][direction] = {}
        if start_time_window not in volumes[tollgate_id][direction]:
            volumes[tollgate_id][direction][start_time_window] = 1
        else:
            volumes[tollgate_id][direction][start_time_window] += 1

        # 把etc加进去
        has_etc = int(each_pass[4])
        if tollgate_id not in etc:
            etc[tollgate_id] = {}
        if direction not in etc[tollgate_id]:
            etc[tollgate_id][direction] = {}
        if start_time_window not in etc[tollgate_id][direction]:
            etc[tollgate_id][direction][start_time_window] = has_etc
        else:
            etc[tollgate_id][direction][start_time_window] += has_etc


    # Step 3: format output for tollgate and direction per time window
    fw = open(out_file_name, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"', '"etc"']) + '\n')
    for tollgate_id in volumes:
        for direction in volumes[tollgate_id]:
            time_window = list(volumes[tollgate_id][direction].keys())
            time_window.sort()
            for time_window_start in time_window:
                time_window_end = time_window_start + timedelta(minutes=20)
                out_line = ','.join(['"' + str(tollgate_id) + '"',
			                     '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                 '"' + str(direction) + '"',
                                 '"' + str(volumes[tollgate_id][direction][time_window_start]) + '"',
                                 '"' + str(etc[tollgate_id][direction][time_window_start]) + '"',
                               ]) + '\n'
                fw.writelines(out_line)
    fw.close()

def main():

    in_file = 'volume(table 6)_test1'
    avgVolume(in_file)

if __name__ == '__main__':
    main()



