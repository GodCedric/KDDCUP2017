# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
产生全时间数据
"""

from datetime import datetime,date,time,timedelta

fw = open('full_time1.csv','w')
fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"']) + '\n')

start_time = datetime(2016,9,19,0,0,0)
end_time = datetime(2016,10,24,23,40,0)
cur_time = start_time

while cur_time <= end_time:
    time_window_start = cur_time
    time_window_end = time_window_start + timedelta(minutes=20)
    out_line = ','.join(['"' + 'A' + '"', '"' + '2' + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + 'A' + '"', '"' + '3' + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + 'B' + '"', '"' + '1' + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + 'B' + '"', '"' + '3' + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + 'C' + '"', '"' + '1' + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + 'C' + '"', '"' + '3' + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"']) + '\n'
    fw.writelines(out_line)

    cur_time = cur_time + timedelta(minutes=20)
fw.close()


fw = open('full_time2.csv','w')
fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"']) + '\n')

start_time = datetime(2016,9,19,0,0,0)
end_time = datetime(2016,10,24,23,40,0)
cur_time = start_time

while cur_time <= end_time:
    time_window_start = cur_time
    time_window_end = time_window_start + timedelta(minutes=20)

    out_line = ','.join(['"' + str(1) + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                         '"' + str(0) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(1) + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                         '"' + str(1) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(2) + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                         '"' + str(0) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(3) + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                         '"' + str(0) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(3) + '"',
                         '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                         '"' + str(1) + '"']) + '\n'
    fw.writelines(out_line)

    cur_time = cur_time + timedelta(minutes=20)
fw.close()


# 为天气生成
fw = open('full_time_forweather.csv','w')
fw.writelines(','.join(['"date"', '"hour"']) + '\n')

start_time = datetime(2016,7,1,0,0,0)
end_time = datetime(2016,10,24,23,40,0)
cur_time = start_time

while cur_time <= end_time:

    cur_date = cur_time.date()

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(0) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(3) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(6) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(9) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(12) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(15) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(18) + '"']) + '\n'
    fw.writelines(out_line)

    out_line = ','.join(['"' + str(cur_date) + '"',
                         '"' + str(21) + '"']) + '\n'
    fw.writelines(out_line)

    cur_time = cur_time + timedelta(days=1)
fw.close()
