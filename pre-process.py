import os
import numpy as np
import tqdm
import csv
import pandas as pd
from datetime import datetime, date
from sklearn.model_selection import train_test_split, KFold

question_path = './questions.csv'
mid = pd.read_csv(question_path)
problems = mid['question_id']
problem2id = {p: i + 1 for i, p in enumerate(problems)}
id2problem = {i+1: p for i, p in enumerate(problems)}
skill = mid['tags']

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


def count(path):
    data = pd.read_csv(path)
    return len(data['correct'])
length = 0

for i in range(1,5000):
     stu_path = './KT1' + '/u' + str(i) +'.csv'
     if os.path.exists(stu_path):
         length += count(stu_path)
print(length)

def getinfo(path):
    s_ = []
    at_ = []
    e_ = []
    it_ = []
    a_ = []
    data = pd.read_csv(path)
    a = data['correct']
    it = data['interval_time']
    e = data['question_id']
    at = data['answer_time']
    for i in at:
        at_.append(int(i))
    for i in a:
        a_.append(int(i))
    for i in it:
        if i > 43200:
            i = 43200
        it_.append(int(i))
    for i in e:
        e_.append(problem2id[i])
    for i in e:
        temp = skill[problem2id[i] - 1].split(';')
        for j in temp:
            s_.append(skill_id[j])
    return  s_,a_,e_,it_,at_

def parse_all_seq():
  all_sequences = []
  for i in range(1,5000):
     stu_path = './KT1' + '/u' + str(i) +'.csv'
     if os.path.exists(stu_path):
         seq = getinfo(stu_path)
         all_sequences.extend([seq])

  return  all_sequences

sequences = parse_all_seq()


# split train data and test data
train_data, test_data = train_test_split(sequences, test_size=.2, random_state=10)
train_data = np.array(train_data)
test_data = np.array(test_data)

def sequences2l(sequences, trg_path):
    with open(trg_path, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write data into file: %s' % trg_path):
            s_seq, a_seq, p_seq, it_seq, at_seq = seq
            seq_len = len(s_seq)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(s) for s in s_seq]) + '\n')
            f.write(','.join([str(a) for a in a_seq]) + '\n')
            f.write(','.join([str(p) for p in p_seq]) + '\n')
            f.write(','.join([str(i) for i in it_seq]) + '\n')
            f.write(','.join([str(a) for a in at_seq]) + '\n')

# split into 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=10)
idx = 0
for train_data_1, valid_data in kfold.split(train_data):
    sequences2l(train_data[train_data_1], 'train' + str(idx) + '.txt')
    sequences2l(train_data[valid_data], 'valid' + str(idx) + '.txt')
    idx += 1

sequences2l(test_data, 'test.txt')
