# -*- coding: utf-8 -*-
#!/usr/bin/env python

from datetime import datetime,date,time

volumes2 = {}
time_reslut_sequence = [time(8, 0, 0)]
time_reslut_sequence.append(time(8, 20, 0))
time_reslut_sequence.append(time(8, 40, 0))
time_reslut_sequence.append(time(9, 0, 0))
time_reslut_sequence.append(time(9, 20, 0))
time_reslut_sequence.append(time(9, 40, 0))
time_reslut_sequence.append(time(17, 0, 0))
time_reslut_sequence.append(time(17, 20, 0))
time_reslut_sequence.append(time(17, 40, 0))
time_reslut_sequence.append(time(18, 0, 0))
time_reslut_sequence.append(time(18, 20, 0))
time_reslut_sequence.append(time(18, 40, 0))
for i in range(3):
    volumes2[i + 1] = {}
for i in range(3):
    for j in range(2):
        volumes2[i + 1][j] = {}
for i in range(3):
    for j in range(2):
        for k in range(7):
            volumes2[i + 1][j][k] = {}
for i in range(3):
    for j in range(2):
        for k in range(7):
            for n in time_reslut_sequence:
                volumes2[i + 1][j][k][n] = {}
a1 = volumes2.keys()
a2 = volumes2[1].keys()
a3 = volumes2[1][0].keys()



d1 = datetime(2017,4,29,8,0,0)
d2 = datetime(2017,4,29,9,0,0)
dtest1 = datetime(2017,4,29,8,0,0)
dtest2 = datetime(2017,4,29,5,0,0)
dtest3 = datetime(2017,4,29,18,0,0)
dtest4 = datetime(2017,4,29,9,0,0)

a1 = d1.weekday()
a2 = d1.time()

dtest = [dtest1,dtest2,dtest3,dtest4]

for dt in dtest:
    if dt <= d2 and dt >= d1:
        print(dt)
