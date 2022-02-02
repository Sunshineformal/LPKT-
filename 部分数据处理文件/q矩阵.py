import os
import csv
import pandas as pd
import numpy as np

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

gamma = 0.003

data = np.zeros((13170,190))
question_path ='./questions.csv'

problems = pd.read_csv(question_path)['question_id']

problem2id = { p: i+1 for i, p in enumerate(problems) }

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i][j] = gamma

def cmpute(question_path):
 with open(question_path) as f:
    f.readline()
    for i, line in enumerate(csv.reader(f, delimiter=",")):
          tags = []
          id = problem2id[line[0]]
          tags = line[5].split(';')
          for j in tags:
              j = int(skill_id[j])
              if(j == -1):
                  continue
              data[id][j] = 1


def save(skillRelationMatrix):
    file = open("q_matrix.csv","w",newline = "", encoding = "utf-8-sig")
    writer = csv.writer(file)
    for eachLine in skillRelationMatrix:
        writer.writerow(eachLine)
    file.close()

cmpute(question_path)
save(data)
# 189 个skill