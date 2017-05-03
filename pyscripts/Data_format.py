# -*- coding: utf-8 -*-
# !/user/bin/evn python

'''
objective:将清洗后的数据转换成要训练模型时的输入数据格式
author:陆韬宇
'''
import math
from datetime import datetime
file_suffix = '.csv'


# 抽取同一时间窗口不同日期的平均消耗时间数据
def dateTravelTime(in_file, in_file_weather, path):
    # 文件名
    out_suffix = '_20min_date_travel_time'
    in_file_name = in_file + file_suffix
    in_file_weather_name = in_file_weather + file_suffix
    out_file_name = in_file.split('_')[0] + out_suffix + file_suffix

    # 定义目标时间窗口列表（时间窗口起始时间）(列表最后一项仅用于'存储日期及对应的att'任务中进行闭开区间集成)
    target_time_window_horizontal = ['08:00:00','08:20:00','08:40:00',\
                                     '09:00:00','09:20:00','09:40:00',\
                                     '17:00:00','17:20:00','17:40:00',\
                                     '18:00:00','18:20:00','18:40:00']
    
    # 定义路线列表
    tra_tollgate_list = ['A-2','A-3',\
                         'B-1','B-3',\
                         'C-1','C-3']
    
    # Step 1: Load trajectories and average travel time(att)
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    att_data = fr.readlines()
    fr.close()

    # load weather data
    fr = open(path + in_file_weather_name, 'r')
    fr.readline()
    wea_data = fr.readlines()
    fr.close()

    
    # Step 2: Create a dictionary to store att for each route in target time windows
    wea = {}
    wea_num = 0
    while wea_num < len(wea_data):
        if in_file.split('_')[0][:4] == 'test':
            temp_sample = wea_data[wea_num].replace('"', '').strip().split(',')
            wea_day = temp_sample[0]
            wea_hour = temp_sample[1]
        else:
            temp_sample = wea_data[wea_num].strip().split(',')
            temp_day = datetime.strptime(temp_sample[0], '%Y/%m/%d')
            wea_day = datetime.strftime(temp_day, '%Y-%m-%d')
            wea_hour = temp_sample[1]
        # 判断路线对应的存储字典是否存在
        
        if  wea_day not in wea.keys():
            wea[wea_day] = {}
        # 判断起始时间窗口是否在任一目标窗口中
        wea[wea_day][wea_hour] = ' '.join(temp_sample[4:])
        wea_num += 1


    att = {}
    att_num = 0
    while att_num < len(att_data):
        temp_sample = att_data[att_num].replace('"', '').split(',')
        # 判断路线对应的存储字典是否存在
        att_route = temp_sample[0] + '-' + temp_sample[1]
        if  att_route not in att.keys():
            att[att_route] = {}
        # 判断起始时间窗口是否在任一目标窗口中
        for each_tw_num in range(len(target_time_window_horizontal)):
            if temp_sample[2][12:20] == target_time_window_horizontal[each_tw_num]:
                # 存储日期及对应的att
                if temp_sample[2][12:20] == '09:40:00':
                    tw = '[09:40:00 10:00:00)'
                elif temp_sample[2][12:20] == '18:40:00':
                    tw = '[18:40:00 19:00:00)'
                else:
                    tw = '[' + target_time_window_horizontal[each_tw_num] + ' ' + \
                        target_time_window_horizontal[each_tw_num+1] + ')' # 形成格式化时间窗口数据（仅包括时、分、秒）
                weather_hour_key = str(math.floor(int(tw[1:3])/3) * 3)
                weather_day_key = temp_sample[2][1:11]
                if tw not in att[att_route]:
                    att[att_route][tw] = [temp_sample[2][1:11] + ' ' + wea[weather_day_key][weather_hour_key] + ' ' + temp_sample[4].strip()]
                else:
                    att[att_route][tw].append(temp_sample[2][1:11] + ' ' + wea[weather_day_key][weather_hour_key] + ' ' + temp_sample[4].strip())

            else:
                continue
        att_num += 1
        
        

    # Step 3: output
    fw = open(path + out_file_name, 'w')
    # 输出结果文件中每一列代表的意思
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window(%H:%M:%S)"', '"date:weather:aveTravelTime"']) + '\n')
    for route in tra_tollgate_list:
        route_time_windows = list(att[route].keys())
        route_time_windows.sort()
        for time_window in route_time_windows:
            date_tt_set = ';'.join(att[route][time_window])
            out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"',
                                 '"' + time_window + '"','"' + date_tt_set + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()
    

# 抽取同一时间窗口不同日期的车流量数据
def dateVolume(in_file, in_file_weather, path):
    # 文件名
    out_suffix = '_20min_date_volume'
    in_file_name = in_file + file_suffix
    in_file_weather_name = in_file_weather + file_suffix
    out_file_name = in_file.split('_')[0] + out_suffix + file_suffix

    # 定义目标时间窗口列表（时间窗口起始时间）(列表最后一项仅用于'存储日期及对应的att'任务中进行闭开区间集成)
    target_time_window_horizontal = ['08:00:00','08:20:00','08:40:00',\
                                     '09:00:00','09:20:00','09:40:00',\
                                     '17:00:00','17:20:00','17:40:00',\
                                     '18:00:00','18:20:00','18:40:00']
    
    # 定义路线列表
    tollgate_direction_list = ['1-0','1-1',\
                                '2-0',\
                                '3-0','3-1']
    
    # Step 1: Load trajectories and average travel time(att)
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    av_data = fr.readlines()
    fr.close()
    

    # load weather data
    fr = open(path + in_file_weather_name, 'r')
    fr.readline()
    wea_data = fr.readlines()
    fr.close()

    
    # Step 2: Create a dictionary to store att for each route in target time windows
    wea = {}
    wea_num = 0
    while wea_num < len(wea_data):
        if in_file.split('_')[0][:4] == 'test':
            temp_sample = wea_data[wea_num].replace('"', '').strip().split(',')
            wea_day = temp_sample[0]
            wea_hour = temp_sample[1]
        else:
            temp_sample = wea_data[wea_num].strip().split(',')
            temp_day = datetime.strptime(temp_sample[0], '%Y/%m/%d')
            wea_day = datetime.strftime(temp_day, '%Y-%m-%d')
            wea_hour = temp_sample[1]
        if  wea_day not in wea.keys():
            wea[wea_day] = {}
        # 判断起始时间窗口是否在任一目标窗口中
        wea[wea_day][wea_hour] = ' '.join(temp_sample[4:])
        wea_num += 1


    av = {}
    av_num = 0
    while av_num < len(av_data):
        temp_sample = av_data[av_num].replace('"', '').split(',')
        # 判断收费站-方向对所对应的存储字典是否存在
        av_toll_dir = temp_sample[0] + '-' + temp_sample[3]
        #print(av_toll_dir)
        if  av_toll_dir not in av.keys():
            av[av_toll_dir] = {}
        # 判断起始时间窗口是否在任一目标窗口中
        for each_tw_num in range(len(target_time_window_horizontal)):
            if temp_sample[1][12:20] == target_time_window_horizontal[each_tw_num]:
                # 存储日期及对应的av
                if temp_sample[1][12:20] == '09:40:00':
                    tw = '[09:40:00 10:00:00)'
                elif temp_sample[1][12:20] == '18:40:00':
                    tw = '[18:40:00 19:00:00)'
                else:
                    tw = '[' + target_time_window_horizontal[each_tw_num] + ' ' + \
                        target_time_window_horizontal[each_tw_num+1] + ')' # 形成格式化时间窗口数据（仅包括时、分、秒）
                #print(tw)
                weather_hour_key = str(math.floor(int(tw[1:3])/3) * 3)
                weather_day_key = temp_sample[1][1:11]        
                if tw not in av[av_toll_dir]:
                    av[av_toll_dir][tw] = [temp_sample[1][1:11] + ' ' + wea[weather_day_key][weather_hour_key] + ' ' + temp_sample[4].strip()]
                else:
                    av[av_toll_dir][tw].append(temp_sample[1][1:11] + ' ' + wea[weather_day_key][weather_hour_key] + ' ' + temp_sample[4].strip())
            else:
                continue
        av_num += 1
        
        

    # Step 3: output
    fw = open(path + out_file_name, 'w')
    # 输出结果文件中每一列代表的意思
    fw.writelines(','.join(['"tollgate_id"', '"direction_id"', '"time_window(%H:%M:%S)"', '"date:weather:aveVolume"']) + '\n')
    for toll_dir in tollgate_direction_list:
        toll_dir_time_windows = list(av[toll_dir].keys())
        toll_dir_time_windows.sort()
        for time_window in toll_dir_time_windows:
            date_v_set = ';'.join(av[toll_dir][time_window])
            out_line = ','.join(['"' + toll_dir.split('-')[0] + '"', '"' + toll_dir.split('-')[1] + '"',
                                 '"' + time_window + '"','"' + date_v_set + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()



# 抽取每一天目标时间窗口前2小时的平均消耗时间数据
def hourTravelTime(in_file, in_file_weather, path):
    # 文件名
    out_suffix = '_20min_hour_travel_time'
    in_file_name = in_file + file_suffix
    in_file_weather_name = in_file_weather + file_suffix
    out_file_name = in_file.split('_')[0] + out_suffix + file_suffix

    # 定义目标时间窗口列表（时间窗口起始时间）(列表最后一项仅用于'存储日期及对应的att'任务中进行闭开区间集成)
    green_time_window_morning = ['06:00:00','06:20:00','06:40:00',\
                                 '07:00:00','07:20:00','07:40:00']
    green_time_window_afternoon = [ '15:00:00','15:20:00','15:40:00',\
                                    '16:00:00','16:20:00','16:40:00']
    
    # 定义路线列表
    tra_tollgate_list = ['A-2','A-3',\
                         'B-1','B-3',\
                         'C-1','C-3']
    
    # Step 1: Load trajectories and average travel time(att)
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    htt_data = fr.readlines()
    fr.close()
    
    # load weather data
    fr = open(path + in_file_weather_name, 'r')
    fr.readline()
    wea_data = fr.readlines()
    fr.close()

    
    # Step 2: Create a dictionary to store att for each route in target time windows
    wea = {}
    wea_num = 0
    while wea_num < len(wea_data):
        if in_file.split('_')[0][:4] == 'test':
            temp_sample = wea_data[wea_num].replace('"', '').strip().split(',')
            wea_day = temp_sample[0]
            wea_hour = temp_sample[1]
        else:
            temp_sample = wea_data[wea_num].strip().split(',')
            temp_day = datetime.strptime(temp_sample[0], '%Y/%m/%d')
            wea_day = datetime.strftime(temp_day, '%Y-%m-%d')
            wea_hour = temp_sample[1]
        if  wea_day not in wea.keys():
            wea[wea_day] = {}
        # 判断起始时间窗口是否在任一目标窗口中
        wea[wea_day][wea_hour] = ' '.join(temp_sample[4:])
        wea_num += 1


    htt = {}
    htt_num = 0
    temp_list = []
    num_temp = 0
    while htt_num < len(htt_data):
        temp_sample = htt_data[htt_num].replace('"', '').split(',')
        # 判断路线对应的存储字典是否存在
        htt_route = temp_sample[0] + '-' + temp_sample[1]
        if  htt_route not in htt.keys():
            htt[htt_route] = {}
        temp_weekday = datetime.strptime(temp_sample[2][1:11], '%Y-%m-%d').weekday()
        if temp_weekday not in htt[htt_route].keys():
            htt[htt_route][temp_weekday] = {}
        # 将上午和下午绿色时隙(各六个时间段)内的数据取出分开放置字典中不同的键后
        if temp_sample[2][12:20] in green_time_window_morning:
            tw = '[06:00:00 08:00:00)'
        elif temp_sample[2][12:20] in green_time_window_afternoon:
            tw = '[15:00:00 17:00:00)'
        else:
            htt_num += 1
            continue
        weather_hour_key = str(math.floor(int(tw[1:3])/3) * 3)
        weather_day_key = temp_sample[2][1:11]  
        if tw not in htt[htt_route][temp_weekday].keys():
            htt[htt_route][temp_weekday][tw] = []
        
        htt[htt_route][temp_weekday][tw].append(temp_sample[2][1:11] + ' ' + wea[weather_day_key][weather_hour_key] + ' ' + temp_sample[4].strip())
        '''
        #将绿色时隙中的时间窗口的数据存入一个暂时的列表，当某一绿色时隙段（6个窗口）都存入列表后，
        #再将该暂时列表传入htt字典对应的位置处
        temp_list.append(temp_sample[2][1:11] + ' ' + temp_sample[4].strip())
        num_temp += 1
        if num_temp == 6:
            htt[htt_route][temp_weekday][tw].append(';'.join(temp_list))
            temp_list = []
            num_temp = 0
        '''
        htt_num += 1
        
    
    # Step 3: output
    fw = open(path + out_file_name, 'w')
    # 输出结果文件中每一列代表的意思
    fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', 'weekday', '"time_window(%H:%M:%S)"', '"[date:weather:aveTravelTime]"']) + '\n')
    for route in tra_tollgate_list:
        route_weekdays = list(htt[route].keys())
        route_weekdays.sort()
        for everyweekday in route_weekdays:
            for time_window in ['[06:00:00 08:00:00)', '[15:00:00 17:00:00)']:
                hour_tt_set = ';'.join(htt[route][everyweekday][time_window])
                out_line = ','.join(['"' + route.split('-')[0] + '"', '"' + route.split('-')[1] + '"', '"' + str(everyweekday+1) + '"',
                                    '"' + time_window + '"','"' + hour_tt_set + '"']) + '\n'
                fw.writelines(out_line)
    fw.close()
    

# 抽取每一天目标时间窗口前2小时的车流量数据
def hourVolume(in_file, in_file_weather, path):
    # 文件名
    out_suffix = '_20min_hour_volume'
    in_file_name = in_file + file_suffix
    in_file_weather_name = in_file_weather + file_suffix
    out_file_name = in_file.split('_')[0] + out_suffix + file_suffix

    # 定义目标时间窗口列表（时间窗口起始时间）(列表最后一项仅用于'存储日期及对应的att'任务中进行闭开区间集成)
    green_time_window_morning = ['06:00:00','06:20:00','06:40:00',\
                                '07:00:00','07:20:00','07:40:00']
    green_time_window_afternoon = [ '15:00:00','15:20:00','15:40:00',\
                                    '16:00:00','16:20:00','16:40:00']
    # 定义路线列表
    tollgate_direction_list = ['1-0','1-1',\
                                '2-0',\
                                '3-0','3-1']
    
    # Step 1: Load trajectories and average travel time(att)
    fr = open(path + in_file_name, 'r')
    fr.readline()  # skip the header
    hav_data = fr.readlines()
    fr.close()
    
    # load weather data
    fr = open(path + in_file_weather_name, 'r')
    fr.readline()
    wea_data = fr.readlines()
    fr.close()

    
    # Step 2: Create a dictionary to store att for each route in target time windows
    wea = {}
    wea_num = 0
    while wea_num < len(wea_data):
        if in_file.split('_')[0][:4] == 'test':
            temp_sample = wea_data[wea_num].replace('"', '').strip().split(',')
            wea_day = temp_sample[0]
            wea_hour = temp_sample[1]
        else:
            temp_sample = wea_data[wea_num].strip().split(',')
            temp_day = datetime.strptime(temp_sample[0], '%Y/%m/%d')
            wea_day = datetime.strftime(temp_day, '%Y-%m-%d')
            wea_hour = temp_sample[1]
        if  wea_day not in wea.keys():
            wea[wea_day] = {}
        # 判断起始时间窗口是否在任一目标窗口中
        wea[wea_day][wea_hour] = ' '.join(temp_sample[4:])
        wea_num += 1

    hav = {}
    hav_num = 0
    temp_list = []
    num_temp = 0
    while hav_num < len(hav_data):
        temp_sample = hav_data[hav_num].replace('"', '').split(',')
        # 判断收费站-方向对所对应的存储字典是否存在
        hav_toll_dir = temp_sample[0] + '-' + temp_sample[3]
        #print(hav_toll_dir)
        if  hav_toll_dir not in hav.keys():
            hav[hav_toll_dir] = {}
        temp_weekday = datetime.strptime(temp_sample[1][1:11], '%Y-%m-%d').weekday()
        if temp_weekday not in hav[hav_toll_dir].keys():
            hav[hav_toll_dir][temp_weekday] = {}
        
        # 判断时间窗口位于上午绿色时隙还是下午绿色时隙
        if temp_sample[1][12:20] in green_time_window_morning:
            tw = '[06:00:00 08:00:00)'
        elif temp_sample[1][12:20] in green_time_window_afternoon:
            tw = '[15:00:00 17:00:00)'
        else:
            hav_num += 1
            continue
        #print(tw)
        weather_hour_key = str(math.floor(int(tw[1:3])/3) * 3)
        weather_day_key = temp_sample[1][1:11]          
        if tw not in hav[hav_toll_dir][temp_weekday].keys():
            hav[hav_toll_dir][temp_weekday][tw] = []

        hav[hav_toll_dir][temp_weekday][tw].append(temp_sample[1][1:11] + ' ' + wea[weather_day_key][weather_hour_key] + ' ' + temp_sample[4].strip())        
        '''
        #将绿色时隙中的时间窗口的数据存入一个暂时的列表，当某一绿色时隙段（6个窗口）都存入列表后，
        #再将该暂时列表传入htt字典对应的位置处
        temp_list.append(temp_sample[1][1:11] + ' ' + temp_sample[4].strip())
        num_temp += 1
        if num_temp == 6:
            hav[hav_toll_dir][temp_weekday][tw].append(';'.join(temp_list))
            temp_list = []
            num_temp = 0
        '''
        hav_num += 1
        
            

    # Step 3: output
    fw = open(path + out_file_name, 'w')
    # 输出结果文件中每一列代表的意思
    fw.writelines(','.join(['"tollgate_id"', '"direction_id"', 'weekday', '"time_window(%H:%M:%S)"', '"[date:weather:aveVolume]"']) + '\n')
    for toll_dir in tollgate_direction_list:
        toll_dir_weekdays = list(hav[toll_dir].keys())
        toll_dir_weekdays.sort()
        for everyweekday in toll_dir_weekdays:
            for time_window in ['[06:00:00 08:00:00)', '[15:00:00 17:00:00)']:
                hour_v_set = ';'.join(hav[toll_dir][everyweekday][time_window])
                out_line = ','.join(['"' + toll_dir.split('-')[0] + '"', '"' + toll_dir.split('-')[1] + '"', '"' + str(everyweekday+1) + '"',
                                    '"' + time_window + '"','"' + hour_v_set + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()


def main():
    #路径定义
    path_train = 'd:/KDD Cup 2017/dataSets/training/DataProcessed/'
    path_test = 'd:/KDD Cup 2017/dataSets/testing_phase1/DataProcessed/'



    #训练数据抽取
    #将转化后的平均消耗时间数据抽取成预期数据格式
    in_file_att = 'training_20min_avg_travel_time'
    in_file_weather = 'weather (table 7)_training_update'
    dateTravelTime(in_file_att, in_file_weather, path_train)
    hourTravelTime(in_file_att, in_file_weather, path_train)
    #将转化后的车流量数据抽取成预期数据格式
    in_file_av = 'training_20min_avg_volume'
    dateVolume(in_file_av, in_file_weather, path_train)
    hourVolume(in_file_av, in_file_weather, path_train)



    #测试数据抽取
    in_file_at_t = 'test1_20min_avg_travel_time'
    in_file_weather_t = 'weather (table 7)_test1'
    hourTravelTime(in_file_at_t, in_file_weather_t, path_test)
    
    in_file_av_t = 'test1_20min_avg_volume'
    hourVolume(in_file_av_t, in_file_weather_t, path_test)



if __name__ == '__main__':
    main()
