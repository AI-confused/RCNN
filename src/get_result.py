import pandas as pd
import csv
import argparse
import numpy as np


   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose test result path')
    parser.add_argument("-model", default=None, type=str, required=True)
    parser.add_argument("-output", default=None, type=str, required=True)
    parser.add_argument("-k", default=None, type=int, required=True)
    args = parser.parse_args()

    df=pd.read_csv('../data/test_clean.csv')
    df['0']=0
    df['1']=0
    df['2']=0
    for i in range(args.k):
        temp=pd.read_csv('../output/10-26/'+str(args.model)+'/fold_{}/test_result_{}.csv'.format(args.k,i))
        df['0']+=temp['label_0']/args.k
        df['1']+=temp['label_1']/args.k
        df['2']+=temp['label_2']/args.k
    print(df['0'].mean())

#     df['label']=np.argmax(df[['0', '1', '2']].values,-1)
#     df[['id','label']].to_csv(args.output,index=False)
    df[['id', '0','1','2']].to_csv(args.output,index=False)