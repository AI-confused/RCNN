import pandas as pd
import numpy as np
import random
import csv
import os
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='k折交叉划分训练集验证集')
    # read outside parameters
    parser.add_argument('-k', type=int, default=5, help='k fold')
    parser.add_argument('-train_file', type=str, default='../data/train_clean.csv', help='input train dev file')
    parser.add_argument('-test_file', type=str, default='../data/test_clean.csv', help='input test file')
    args = parser.parse_args()
    # read train_clean file
    train = pd.read_csv(args.train_file, index_col=False, encoding='utf-8')
    # read test_clean file
    test = pd.read_csv(args.test_file, index_col=False, encoding='utf-8')
    index=set(range(train.shape[0]))
    K_fold=[]
    # split to k fold
    for i in range(args.k):
        if i == args.k-1:
            tmp=index
        else:
            tmp=random.sample(index,int(1.0/args.k*train.shape[0]))
        index=index-set(tmp)
        print("Number:",len(tmp))
        K_fold.append(tmp)

    os.system('mkdir ../data/fold_{}/'.format(args.k))
    for i in range(args.k):
        print("Fold",i)
        os.system("mkdir ../data/fold_{}/data_{}".format(args.k,i))
        dev_index=list(K_fold[i])
        train_index=[]
        for j in range(args.k):
            if j!=i:
                train_index+=K_fold[j]
        train.iloc[train_index].to_csv("../data/fold_{}/data_{}/train.csv".format(args.k,i), index=False)
        train.iloc[dev_index].to_csv("../data/fold_{}/data_{}/dev.csv".format(args.k,i), index=False)
        test.to_csv("../data/fold_{}/data_{}/test.csv".format(args.k,i), index=False)
