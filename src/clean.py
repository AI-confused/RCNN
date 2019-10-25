import codecs
import pandas as pd
import numpy as np
import argparse
import jieba
import os

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def is_punctuation(uchar):
    punctuations = ['，', '。', '？', '！', '：']
    if uchar in punctuations:
        return True
    else:
        return False

    
    
def text_process(text):
    temp_text = ''
    for _ in text:
        if is_chinese(_) or is_punctuation(_):
            temp_text += _
    text = temp_text
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean data')
    # read outside parameters
    parser.add_argument('-write-train', type=str, default='../data/train_clean.csv', help='write train file name')
    parser.add_argument('-write-test', type=str, default='../data/test_clean.csv', help='write test file name')
#     parser.add_argument('-test', type=int, default=0, help='clean test file')
#     parser.add_argument('-train', type=int, default=1, help='clean train file')
    args = parser.parse_args()
    
    train=pd.read_csv("../data/Train_DataSet.csv")
    train_label_df=pd.read_csv("../data/Train_DataSet_Label.csv")
    test=pd.read_csv("../data/Test_DataSet.csv")
    train=train.merge(train_label_df,on='id',how='left')
    train['label']=train['label'].fillna(-1)
    train=train[train['label']!=-1]
    train['label']=train['label'].astype(int)

    test['content']=test['content'].fillna('无')
    train['content']=train['content'].fillna('无')
    test['title']=test['title'].fillna('无')
    train['title']=train['title'].fillna('无')
    
    # clean train file
    for i in train.index:
        # clean title
        title = text_process(str(train.loc[i, 'title']))
        if title == '':
            train.loc[i, 'title'] = '无'
        else:
            train.loc[i, 'title'] = title
        # clean content
        content = text_process(str(train.loc[i, 'content']))
        if content == '':
            train.loc[i, 'content'] = '无'
        else:
            train.loc[i, 'content'] = content
    # write to new csv
    train.to_csv(args.write_train, index=False, encoding='utf-8')
    print('clean train done')
    # clean test file
    for i in test.index:
        title = text_process(str(test.loc[i, 'title']))
        if title == '':
            test.loc[i, 'title'] = '无'
        else:
            test.loc[i, 'title'] = title
        content = text_process(str(test.loc[i, 'content']))
        if content == '':
            test.loc[i, 'content'] = '无'
        else:
            test.loc[i, 'content'] = content
    # write to new csv
    test.to_csv(args.write_test, index=False, encoding='utf-8')
    print('clean test done')
