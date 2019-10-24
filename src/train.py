import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import data
import datetime
from torch.autograd import Variable
import numpy as np
import csv

class ConfMatrix(object):
    def __init__(self, num_label):
        self.conf = np.zeros((num_label,num_label), dtype=int)
        
    def change_conf(self, i, j):
        self.conf[i][j] += 1
          
    def get_macro_f1(self, label, num_label):
        eplison = 1e-8
        pre_sum = 0
        recall_sum = 0
        for _ in range(num_label):
            pre_sum += self.conf[_][label]
            recall_sum += self.conf[label][_]
        prediction = float(self.conf[label][label]/(pre_sum+eplison))
        recall = float(self.conf[label][label]/(recall_sum+eplison))
        macro_f1 = float((2*prediction*recall) / (prediction + recall + eplison))
        return macro_f1
        
    def get_average_macro_f1(self, macro_f1_list):
        sum = 0
        for _ in macro_f1_list:   
            sum += _
        return float(sum/len(macro_f1_list))
    
    

def train(model, **kargs):
    model.train()
    accu_loss = 0
    for _ in range(len(kargs['train'])):   
        output = model(x=Variable(torch.FloatTensor(kargs['train'][_])).to(kargs['device']), h0=Variable(torch.zeros((2, len(kargs['train'][_]), kargs['hidden_size']))).to(kargs['device']), seq_len=kargs['seq_len'], input_size=kargs['input_size'], hidden_size=kargs['hidden_size'], linear_size=100)# batch * 3
        label = np.array(kargs['train_label'][_]) 
        label = Variable(torch.LongTensor(label)).to(kargs['device'])
        loss = kargs['loss_func'](output, label)
#         if kargs['n_gpu'] > 1:
#             loss = loss.mean() # average on multi-gpu
        accu_loss += loss.item()

        kargs['optimizer'].zero_grad()
        loss.backward()
        kargs['optimizer'].step()
    now = datetime.datetime.now()
    with open(kargs['eval_result'], 'a') as f:
        f.write('['+str(now)+']:epoch '+str(kargs['epoch'])+' loss: '+str(accu_loss/len(kargs['train']))+'\n')
    print('['+str(now)+']:epoch '+str(kargs['epoch'])+' loss: '+str(accu_loss/len(kargs['train'])))


def eval(model, **kargs):
    model.eval()
    right_count = 0
    confmatrix = ConfMatrix(kargs['num_label'])
    for _ in range(len(kargs['dev'])):
        output = model.forward(x=Variable(torch.FloatTensor(kargs['dev'][_])).to(kargs['device']), h0=Variable(torch.zeros((2, len(kargs['dev'][_]), kargs['hidden_size']))).to(kargs['device']), seq_len=kargs['seq_len'], input_size=kargs['input_size'], hidden_size=kargs['hidden_size'], linear_size=100)
        output = output.max(1)[1]
        for y, label in enumerate(kargs['dev_label'][_]):
            if output.cpu().numpy()[y] == label:
                right_count += 1
            confmatrix.change_conf(label, output.cpu().numpy()[y])
    accuracy = float(right_count/(kargs['batch_size']*(len(kargs['dev'])-1)+len(kargs['dev'][-1])))
    macro_f1_list = []
    for j in range(kargs['num_label']):
        macro_f1_list.append(confmatrix.get_macro_f1(j, kargs['num_label']))
    macro_f1 = confmatrix.get_average_macro_f1(macro_f1_list)
    now = datetime.datetime.now()
    with open(kargs['eval_result'], 'a') as f:
        f.write('['+str(now)+']:epoch '+str(kargs['epoch'])+': accuracy: '+str(accuracy) + '| macro f1: ' + str(macro_f1)+'\n')
        f.write('*'*30+'\n')
    print('['+str(now)+']:epoch '+str(kargs['epoch'])+': accuracy: '+str(accuracy) + '| macro f1: ' + str(macro_f1))
    return macro_f1

def write_csv(content, csv_file):
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(content)
    
    
    
def predict(model, test, device, file):
    model.eval()
    write_csv(['label_0', 'label_1', 'label_2'], file)
    for _ in range(len(test)):
        output = model.forward(Variable(torch.FloatTensor(test[_])).to(device)).detach()
        output = output.cpu().numpy().tolist()
        for item in output:
            write_csv(item, file)

