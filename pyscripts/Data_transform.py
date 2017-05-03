# -*- coding: utf-8 -*-
# !/user/bin/evn python

'''
objective:
get clean dataset from collected data
'''

from datetime import datetime, timedelta
import math
file_suffix = '.csv'

# 初始数据转换为平均消耗时间
def avgTravelTime(in_file, path_in, path_out):

    out_suffix = '_20min_avg_travel_time'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Step 1: Load trajectories
    fr = open(path_in + in_file_name, 'r')
    fr.readline()  # skip the header
    traj_data = fr.readlines()
    fr.close()
    #print(traj_data[0])

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
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = math.floor(trace_start_time.minute / 20) * 20
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
        tt = float(each_traj[-1]) # travel time

        #将位于同一个开始时间窗口内所有样本的travel_time放入对应的子字典中
        if start_time_window not in travel_times[route_id].keys():
            travel_times[route_id][start_time_window] = [tt]
        else:
            travel_times[route_id][start_time_window].append(tt)

    # 定义路线列表
    tra_tollgate_list = ['A-2','A-3',\
                         'B-1','B-3',\
                         'C-1','C-3']
    # Step 3: Calculate average travel time for each route per time window
    fw = open(path_out + out_file_name, 'w')
    # 输出结果文件中每一列代表的意思
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
    for route in tra_tollgate_list:
        route_time_windows = list(travel_times[route].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            time_window_end = time_window_start + timedelta(minutes=20)
            tt_set = travel_times[route][time_window_start]
            avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)#round函数进行四舍五入，精度是小数点后两位
            out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                                 '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                                 '"' + str(avg_tt) + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()

# 初始数据转换成车流量
def avgVolume(in_file, path_in, path_out):

    out_suffix = '_20min_avg_volume'
    in_file_name = in_file + file_suffix
    out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

    # Step 1: Load volume data
    fr = open(path_in + in_file_name, 'r')
    fr.readline()  # skip the header
    vol_data = fr.readlines()
    #print(vol_data[0])
    fr.close()

    # Step 2: Create a dictionary to caculate and store volume per time window
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


    # 定义路线列表
    tollgate_direction_list = ['1-0','1-1',\
                                '2-0',\
                                '3-0','3-1']

    # Step 3: format output for tollgate and direction per time window
    fw = open(path_out + out_file_name, 'w')
    fw.writelines(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    time_windows = list(volumes.keys())
    time_windows.sort()
    for time_window_start in time_windows:
        time_window_end = time_window_start + timedelta(minutes=20)
        for tollgate_id in volumes[time_window_start].keys():
            for direction_id in volumes[time_window_start][tollgate_id].keys():
                out_line = ','.join(['"' + str(tollgate_id) + '"', 
			            '"[' + str(time_window_start) + ',' + str(time_window_end) + ')"',
                        '"' + str(direction_id) + '"',
                        '"' + str(volumes[time_window_start][tollgate_id][direction_id]) + '"',
                                    ]) + '\n'
                fw.writelines(out_line)
    fw.close()



def main():
    
    #定义训练数据和测试数据的输入路径与输出路径
    path_train_in = 'd:/KDD Cup 2017/dataSets/training/'
    path_train_out = 'd:/KDD Cup 2017/dataSets/training/DataProcessed/'  # set the data directory
    path_test_in = 'd:/KDD Cup 2017/dataSets/testing_phase1/'
    path_test_out = 'd:/KDD Cup 2017/dataSets/testing_phase1/DataProcessed/'


    #训练数据转换
    #将原始数据转换成用于预测的平均消耗时间
    in_file_tt_o = 'trajectories(table 5)_training'
    avgTravelTime(in_file_tt_o, path_train_in, path_train_out)

    #将原始数据转换成用于预测的车流量
    in_file_volume_o = 'volume(table 6)_training'
    avgVolume(in_file_volume_o, path_train_in, path_train_out)



    #测试数据转换
    #将原始数据转换成用于预测的平均消耗时间
    in_file_tt_t = 'trajectories(table 5)_test1'
    avgTravelTime(in_file_tt_t, path_test_in, path_test_out)

    #将原始数据转换成用于预测的车流量
    in_file_volume_t = 'volume(table 6)_test1'
    avgVolume(in_file_volume_t, path_test_in, path_test_out)


    

if __name__ == '__main__':
    main()
