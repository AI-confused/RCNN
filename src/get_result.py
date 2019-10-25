import pandas as pd
import csv
import argparse
import numpy as np


   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose test result path')
    parser.add_argument("-output", default=None, type=str, required=True)
    parser.add_argument("-k", default=None, type=int, required=True)
    args = parser.parse_args()

    df=pd.read_csv('../data/test_clean.csv')
    df['1']=0
    df['2']=0
    df['3']=0
    df['4']=0
    for i in range(args.k):
        temp=pd.read_csv('../output/model_textcnn/fold_{}/test_result_{}.csv'.format(args.k,i))
        df['1']+=temp['label_1']/args.k
        df['2']+=temp['label_2']/args.k
        df['3']+=temp['label_3']/args.k
        df['4']+=temp['label_4']/args.k
    print(df['1'].mean())

    df['label']=np.argmax(df[['1','2','3', '4']].values,-1)
    for _ in df.index:
        df.loc[_, 'label'] += 1
    df[['id','label']].to_csv(args.output,index=False)
    n1=n2=n3=n4=0
    for _ in df.index:
        if df.loc[_, 'label'] == 1:
            n1 += 1
        elif df.loc[_, 'label'] == 2:
            n2 += 1
        elif df.loc[_, 'label'] == 3:
            n3 += 1
        else:
            n4 += 1
    print([n1,n2,n3,n4])
