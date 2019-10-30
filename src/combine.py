import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-k", default=None, type=int, required=True)
parser.add_argument("-model", default=None, type=str, required=True)
parser.add_argument("-output", default=None, type=str, required=True)
args = parser.parse_args()

k=args.k
df=pd.read_csv('../data/submit_example.csv')
df['0']=0
df['1']=0
df['2']=0
for i in range(k):
    temp=pd.read_csv('../output/{}{}.csv'.format(args.model,i))
    df['0']+=temp['0']/k
    df['1']+=temp['1']/k
    df['2']+=temp['2']/k
print(df['0'].mean())
 
df['label']=np.argmax(df[['0','1','2']].values,-1)
df[['id','label']].to_csv(args.output,index=False)
# df[['id', '0','1','2']].to_csv(args.out_path,index=False)