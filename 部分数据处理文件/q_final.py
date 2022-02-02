import pandas as pd
import csv
import numpy as np


data1 = np.loadtxt('q_martix(final).csv',delimiter=',')
data2 = np.loadtxt('q_matrix.csv',encoding='utf-8',delimiter=',')


for i in range (data2.shape[0]):
    tag = False
    for j in range(data2.shape[1]):
        if data2[i][j] == 1 and  tag == False:
           tag = True
           for m in range(data2.shape[1]):
               if data2[i][m] == 1:
                   continue
               else:
                  data2[i][m] = data2[i][m]*data1[j][m]
    tag = False


def save(skillRelationMatrix):
    file = open("final.csv","w",newline = "", encoding = "utf-8-sig")
    writer = csv.writer(file)
    for eachLine in skillRelationMatrix:
        writer.writerow(eachLine)
    file.close()

print (data2)
save(data2)
