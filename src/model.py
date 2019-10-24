import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



class RCNN_Text(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes, linear_size1, linear_size2, linear_size3, dropout):
        super(RCNN_Text, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.linear_size1 = linear_size1
        self.linear_size2 = linear_size2
        self.linear_size3 = linear_size3
        self.dropout = dropout
        
        self.birnn = nn.RNN(self.input_size, self.hidden_size, bias=False, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(self.linear_size1, self.linear_size2)
        self.linear2 = nn.Linear(self.linear_size3, num_classes)
        self.drop_out = nn.Dropout(self.dropout)


    def forward(self, **kargs):
        output, hn = self.birnn(kargs['x'], kargs['h0']) # output:batch*seq_len*(hidden*2)   hn:2*batch*768
        print(output.shape, hn.shape)
        output = torch.cat((output, kargs['x']), 2) # batch*seq_len*(5*768)
        print(output.shape)
        output = output.view((-1, kargs['input_size']+2*kargs['hidden_size']))
        print(output.shape)
        output = self.drop_out(output)
        print(output.shape)
        output = self.linear1(output) # batch*seq_len*768
        print(output.shape)
        output = output.view((-1, kargs['seq_len'], kargs['linear_size']))
        print(output.shape)
        output = output.permute(0, 2, 1)
        print(output.shape)
        output = F.max_pool1d(output, output.size(2)).permute(0, 2, 1).squeeze(1) # batch*768
        print(output.shape)
        output = self.drop_out(output)
        logit = self.linear2(output)
        print('logit:')
        print(logit.shape)
        
        return logit
