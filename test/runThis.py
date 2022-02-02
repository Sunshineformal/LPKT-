# coding: utf-8
# 2021/8/17 @ sone

from LPKT import LPKT
import pandas as pd
import numpy as np
import csv
from load_data import DATA
import torch.nn as nn
batch_size = 64
n_at = 18576551
n_it = 18576551
n_question = 188
n_exercise = 18576551
seqlen = 50
d_k = 128
d_a = 128
d_e = 128
dropout = 0.2


path = './q_matrix.csv'
data = pd.read_csv(path)
data = np.array(data,dtype= float)
q_matrix = data


dat = DATA(seqlen=seqlen, separate_char=',')

#logging.getLogger().setLevel(logging.INFO)

# k-fold cross validation
k, train_auc_sum, valid_auc_sum = 5, .0, .0
for i in range(k):
    lpkt = nn.DataParallel(LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout))
    train_data = dat.load_data('./train' + str(i) + '.txt')
    print(train_data)
    valid_data = dat.load_data('./valid' + str(i) + '.txt')
    best_train_auc, best_valid_auc = lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)
    print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
    train_auc_sum += best_train_auc
    valid_auc_sum += best_valid_auc
print('%d-fold validation: avg of best train auc %f, avg of best valid auc %f' % (k, train_auc_sum / k, valid_auc_sum / k))



# train and pred
train_data = dat.load_data('./train0.txt')
valid_data = dat.load_data('./valid0.txt')
test_data = dat.load_data('./test.txt')

lpkt = nn.DataParallel(LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout))
lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)

lpkt.save("lpkt.params")

# lpkt.load("lpkt.params")

_, auc, accuracy = lpkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
