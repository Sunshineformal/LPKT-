import os
import csv
import pandas as pd

def rea(file):
    list = []
    Data = pd.read_csv(file)
    time1 = Data['timestamp']
    time2 = Data['elapsed_time']
    list2 = []
    list.append(0)
    for i in range(1,len(time1)):
        mid = (time1[i]-time1[i-1])/1000/60
        list.append(mid)
    for i in time2:
        list2.append(i/1000)
    Data['interval_time']  = list
    Data['answer_time'] = list2
    Data.to_csv(file, index=False)


for i in range(1,840474):
    stu_path = './KT1' + '/u' + str(i) +'.csv'
    if os.path.exists(stu_path):
        rea(stu_path)
        print(stu_path)