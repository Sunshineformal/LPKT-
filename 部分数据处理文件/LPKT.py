# coding: utf-8
# 2021/8/17 @ sone

import math
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm
import pandas as pd
from meta import KTM
from LPKTNet import LPKTNet
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train_one_epoch(net, optimizer, criterion, batch_size, a_data, e_data, it_data, at_data):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size))
    shuffled_ind = np.arange(e_data.shape[0])
    np.random.shuffle(shuffled_ind)
    e_data = e_data[shuffled_ind]
    at_data = at_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    pred_list = []
    target_list = []

    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size,:]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size,:]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size,:]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size,:]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        # print(input_e.shape)

        pred = net(input_e, input_at, target, input_it)

        mask = input_e[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        loss = criterion(masked_pred, masked_truth).sum()

        loss.backward()
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        pred_list.append(masked_pred)
        target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


def test_one_epoch(net, batch_size, a_data, e_data, it_data, at_data):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []

    for idx in tqdm.tqdm(range(n), 'Testing'):
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        with torch.no_grad():
            pred = net(input_e, input_at, target, input_it)
            print(target)
            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


class LPKT(KTM):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout=0.2):
        super(LPKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        self.lpkt_net = LPKTNet(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, dropout).to(device)
        self.batch_size = batch_size

    def train(self, train_data, test_data=None, *, epoch=10, lr=0.002, lr_decay_step=15, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.lpkt_net.parameters(), lr=lr, eps=1e-8, betas=(0.1, 0.999), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')
        best_train_auc, best_test_auc = .0, .0

        for idx in range(epoch):
            train_loss, train_auc, train_accuracy = train_one_epoch(self.lpkt_net, optimizer, criterion,self.batch_size, *train_data)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            if train_auc > best_train_auc:
                best_train_auc = train_auc

            scheduler.step()

            if test_data is not None:
                test_loss, test_auc, test_accuracy = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, test_auc, test_accuracy))
                if test_auc > best_test_auc:
                    best_test_auc = test_auc

        return best_train_auc, best_test_auc

    def eval(self, test_data) -> ...:
        self.lpkt_net.eval()
        return test_one_epoch(self.lpkt_net, self.batch_size, *test_data)

    def save(self, filepath) -> ...:
        torch.save(self.lpkt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.lpkt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
    def forward(self,e_data, at_data, a_data, it_data):
         return

path = './q_Matrix.csv'
data = pd.read_csv(path)
data = np.array(data,dtype= float)
q_matrix = data

question_path = './questions.csv'
problems = pd.read_csv(question_path)['question_id']
problem2id = {p: i + 1 for i, p in enumerate(problems)}

def getinfo(path,at_,e_,it_,a_):

    data = pd.read_csv(path)
    a = data['correct']
    it = data['interval_time']
    e = data['question_id']
    at = data['answer_time']

    for i in at:
        at_.append(i)
    for i in a:
        a_.append(i)
    for i in it:
        it_.append(i)
    for i in e:
        e_.append(problem2id[i])

at_data =[]
e_data = []
it_data = []
a_data = []

for i in range(1,2):
    stu_path = './KT1' + '/u' + str(i) +'.csv'
    if os.path.exists(stu_path):
         getinfo(stu_path,at_data,e_data,it_data,a_data)


n_question = 13169
n_at = len(at_data)
n_a = len(a_data)
n_it = len(it_data)
n_exercise = len(e_data)
d_a = 64
d_e = 64
d_k = 64
batch_size = 64
seqlen = 50
module = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size)

a_dataArray = np.zeros((len(a_data),seqlen))
for j in range(len(a_data)):
    dat = a_data[j]
    a_dataArray[j, :len(dat)] = dat

e_dataArray = np.zeros((len(e_data),seqlen))
for j in range(len(e_data)):
    dat = e_data[j]
    e_dataArray[j, :len(dat)] = dat

it_dataArray = np.zeros((len(it_data),seqlen))
for j in range(len(it_data)):
    dat = it_data[j]
    it_dataArray[j, :len(dat)] = dat

at_dataArray = np.zeros((len(at_data),seqlen))
for j in range(len(at_data)):
    dat = at_data[j]
    at_dataArray[j, :len(dat)] = dat

train_data = [a_dataArray, e_dataArray,it_dataArray,at_dataArray]



module.train(train_data)