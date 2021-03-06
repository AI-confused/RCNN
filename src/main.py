import os
import argparse
import datetime
import torch
from bert_serving.client import BertClient
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm, trange
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import csv
import json
import data
import model
import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RCNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate [default: 0.001]')
#     parser.add_argument('-per-gpu-train-batch-size', type=int, default=2, help='per gpu batch size for training [default: 2]')
#     parser.add_argument('-per-gpu-eval-batch-size', type=int, default=32, help='per gpu batch size for eval [default: 32]')
    parser.add_argument('-batch-size', type=int, default=32, help='batch size for train [default: 32]')
    parser.add_argument('-save', type=int, default=1, help='whether to save model')
    parser.add_argument('-model', type=str, default=None, required=True, help='dir to store model')
    parser.add_argument('-predict-file', type=str, default=None, required=True, help='dir to store predict')
    # data 
    parser.add_argument('-data-dir', type=str, default='../data', help='dataset dir')
    parser.add_argument('-train-file', type=str, default=None, required=True, help='train file')
    parser.add_argument('-dev-file', type=str, default=None, required=True, help='dev file')
    parser.add_argument('-test-file', type=str, default=None, required=True, help='test file')
    parser.add_argument('-eval-result', type=str, default=None, required=True, help='eval result file')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-input-size', type=int, default=768, help='input dimensiton')
    parser.add_argument('-hidden-size', type=int, default=100, help='rnn hidden layer size')
    parser.add_argument('-seq-len', type=int, default=400, help='input seq len')
    parser.add_argument('-hidden_layers', type=int, default=1, help='rnn hidden layers')
    parser.add_argument('-directions', type=int, default=2, help='bidirection rnn')
    parser.add_argument('-concat', type=int, default=None, required=True, help='title&content concat type')
    parser.add_argument('-linear-size', type=int, default=100, help='linear1 dimensiton')
    parser.add_argument('-train-steps', type=int, default=100, help='train steps, one step for one batch')
    parser.add_argument('-model-type', type=str, default='rnn', help='recurrent network model type')
    # device
    parser.add_argument('-server-ip', type=str, default=None, required=True, help='device ip for bert-as-service server')
    parser.add_argument('-do-predict', type=int, default=0, help='predict the sentence given')
    parser.add_argument('-do-train', type=int, default=1, help='train or test')
    parser.add_argument('-load-model', type=str, default=None, required=True, help='predict loading model dir')
    parser.add_argument('-port', type=int, default=5555, help='bert-as-service port')
    parser.add_argument('-port-out', type=int, default=5556, help='bert-as-service port')
    args = parser.parse_args()
    
    # connect to bert-serving-server
    bc = BertClient(ip=args.server_ip,port=args.port, port_out=args.port_out, check_version=False, check_length=False)
    mydata = data.MyTaskProcessor()
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)
    # set-up cuda,gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # update args
    args.class_num = len(mydata.get_labels()) # 3
    
    # define text rcnn model 
    print('lr: '+str(args.lr))
    rcnn = model.RCNN_Text(args.input_size, args.hidden_size, args.class_num, args.input_size+2*args.hidden_size, args.linear_size, args.dropout, args.model_type).to(device) #need to modify
    
    # training
    if args.do_train:
        
        # open eval_result.txt   
        with open(args.eval_result, 'w') as f:
            now = datetime.datetime.now()
            f.write('['+str(now)+']\n')
            f.write('*'*80+'\n')
            
        # get eval data&label
        dev, dev_label = mydata.get_dev_examples(args.data_dir, args.batch_size, bc, args.dev_file, args.concat)
        train_, train_label = mydata.get_train_examples(args.data_dir, args.batch_size, bc, args.train_file, args.concat)
        print('='*80)
        logger.info("*** Start Training ***")
        logger.info("  Num examples = %d", (len(train_)-1)*args.batch_size+len(train_[-1]))
        logger.info("  Batch size = %d", len(train_))
        logger.info("  Num steps = %d", args.train_steps)
        
        # set tqdm bar
        num_train_optimization_steps = args.train_steps
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        
        # loss_func define
        loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.19462648, 0.25, 0.310986013])).float(), size_average=True).to(device)
        optimizer = torch.optim.Adam(rcnn.parameters(), lr=args.lr)   
        max_f1 = 0

        # train progress
        for i, step in enumerate(bar):
            # get batch data
            batch_train = train_[int(i%len(train_))]
            batch_label = train_label[int(i%len(train_label))]
            # train
            train_loss = train.train(rcnn, train=batch_train, train_label=batch_label, loss_func=loss_func, optimizer=optimizer, device=device, eval_result=args.eval_result, hidden_size=args.hidden_size, seq_len=args.seq_len, input_size=args.input_size, linear_size=args.linear_size, step=i, model_type=args.model_type)
            bar.set_description("loss {}".format(train_loss))
            
            # eval
            macro_f1 = train.eval(rcnn, dev=dev, dev_label=dev_label, batch_size=args.batch_size, device=device, num_label=args.class_num, eval_result=args.eval_result, hidden_size=args.hidden_size, seq_len=args.seq_len, input_size=args.input_size, linear_size=args.linear_size, step=i, model_type=args.model_type)
            
            if macro_f1 >= max_f1:
                print("Best F1", macro_f1)
                print("Saving Model......")
                max_f1=macro_f1
                # Save a trained model
                if args.save == 1:
                    torch.save(rcnn.state_dict(), args.model) #保存网络参数
                    print("="*80)
            else:
                print("="*80)
        logger.info("  Max Macro_f1 = %f", max_f1)
        print('='*80)
        with open(args.eval_result, 'a') as f:
            now = datetime.datetime.now()
            f.write('['+str(now)+'] max_macro_f1:' + str(max_f1))
            
    # predicting
    if args.do_predict:
        rcnn.load_state_dict(torch.load(args.load_model))
        print('model load done')
        test = mydata.get_test_examples(args.data_dir, args.batch_size, bc, args.test_file, args.concat)
        print('test data done')
        train.predict(rcnn, test=test, device=device, file=args.predict_file, hidden_size=args.hidden_size, seq_len=args.seq_len, input_size=args.input_size, linear_size=args.linear_size, model_type=args.model_type)
        print('predict saved')
        print('='*80)
    bc.close()    