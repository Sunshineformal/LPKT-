import numpy as np
import math as m
import pandas as pd
import csv
import os

skill_data = pd.read_csv('questions.csv')
skills0 = skill_data['tags']
skills = []
problems0 = skill_data['question_id']

pro2skill = {}

for i in range(len(skills0)):
   for j in skills0[i].split(';'):
       skills.append(j)

for i in range(len(skills0)):
    pro2skill[problems0[i]] = skills0[i]


skill_id = {}
id_skill = {}

sum = 1
id_skill[0] = 'UNK'
id_skill[1] = 'UNK'

for i in skills:
    tag = True
    for j in range(sum):
        if(id_skill[j] == i):
            tag = False
    if(tag == True):
        skill_id[i] = sum
        id_skill[sum] = i
        sum += 1

def tanh(num):
    if(num > 10.00):
        return 1
    ans = m.exp(num) - m.exp(-num)
    ans = ans/(m.exp(num) + m.exp(-num))
    return ans

def curveFit(num):
    if num < 4 and num > 0.1:
        #fit by matlab
        result = 0.8*num**0.35 - 0.08 * num
    elif num <= 0.1:
        result = 0
    elif num >=4:
        result = 1
    if result > 0.5:
        return result
    else:
        return 0


def cal(a,b,c,d):
    n = a+b+c+d
    if (a + c == 0 or b + d == 0):
        return 0
    ans = curveFit(n*(a*d-b*c)*(a*d-b*c)/(a+b)/(c+d)/(a+c)/(b+d))
    return ans


skill2y1 = [0 for i in range(sum)]
skill2y0 = [0 for i in range(sum)]
tot_skill = [0 for i in range(sum)]



for i in range(1,840474):
    stu_path = './KT1' + '/u' + str(i) +'.csv'
    if os.path.exists(stu_path):
        data = pd.read_csv(stu_path)
        correct = data['correct']
        rows = len(correct)
        all_problem = data['question_id']
        for p in range(rows):
            row_skill = pro2skill[all_problem[p]].split(';')
            for j in row_skill:
             skill_j = skill_id[j]
             right = correct[p]
             tot_skill[skill_j] += 1
             if(right == 1):
                skill2y1[skill_j] += 1
             else:
                skill2y0[skill_j] += 1
    print(stu_path)

skillRelationMatrix = np.zeros((sum,sum))

for i in range(sum):
    if (tot_skill[i] > 0):
        for j in range(sum):
            if (tot_skill[j] > 0):
             skillRelationMatrix[i][j] = cal(skill2y1[i],skill2y0[i],skill2y1[j],skill2y0[j])


def save(skillRelationMatrix):
    file = open("relationFromK2.csv","w",newline = "", encoding = "utf-8")
    writer = csv.writer(file)
    for eachLine in skillRelationMatrix:
        writer.writerow(eachLine)
    file.close()

save(skillRelationMatrix)