import pandas as pd
import operator
import collections
from collections import Counter
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import argparse

import time as tme

def parse_data(data_path):
    df_data=pd.read_csv(data_path, delimiter='\t', header=None)
    df_data=df_data.set_axis(['start', 'end', 'sentence','mention_type','information'], axis=1, inplace=False)
    
    #extract the named entity using the two feature 'end' and 'start' 
    m=[]
    for i in df_data.index:
        l=df_data.loc[i]
        sent=word_tokenize(l.sentence)
        s=''
    #     print(i)
        for j in range(int(l.end)-int(l.start)):
             #in some cases we get a wrong end which lead us to an index out of range error 
            if j+int(l[0])<len(sent):
                s+=sent[j+int(l[0])]+' ' 
        m.append(s)
    
    df_data['entity']=m
    #transform mention types to a list of mentions
    df_data.mention_type=df_data.mention_type.apply(lambda x: x.split())
#     df_data.sentence=df_data.sentence.apply(lambda x : '<PAD> '+x+' <PAD>')
    return df_data


def clear_mention(data):
    
    data=data.drop('information',axis=1)
    mention_type_column=[]
    mention_1_column=[]
    for i in data.index:
        out_type = []
        t=[]
        types=data.loc[i].mention_type
        for a in types:
            flag = True
            for b in types:
                if len(a) >= len(b):
                    continue
                if (a == b[:len(a)]) and (b[len(a)] == "/"):
                    flag = False
            if flag:
                out_type.append(a)
        mention_type_column.append(out_type)
        
        l1=[]

        for k in types:
            l1+=k.split('/')
        d=dict(Counter(l1))
        del d['']
        m=max(d.items(), key=operator.itemgetter(1))[0]
        ok=True
        for a in out_type:
            if (m == a[1:len(m)+1]) and (ok==True):
                ok=False
                t.append(a)
        mention_1_column.append(t)
                
                
    data['one_mention']=mention_1_column
    data['clear_mention']=mention_type_column
    return data

#the test and the validation data doesn't have the same distribution as training data 
#and most of the tags are just one level that's i try to split the training data into train/test

def convert(new_data):
    tokens_ls=[]
    sent_id_ls=[]
    tags_ls=[]
    for l in range(len(new_data)):   
        if l%100==0:
            print(l)
        tokens=word_tokenize(new_data[l][0])
        tags=['O' for i in range(len(tokens))]
        for i in range(len(tags)):
            for j in range(1,len(new_data[l])):
                tags[new_data[l][j][0][0]]='B-'+new_data[l][j][1][0]
                if new_data[l][j][0][1]<len(tokens):
                    end=new_data[l][j][0][1]
                else :
                    end=len(tokens)
                for k in range(new_data[l][j][0][0]+1,end):

                    tags[k]='I-'+new_data[l][j][1][0]
        sent_id=[l for i in range(len(tokens))]     
        tokens_ls+=tokens
        tags_ls+=tags
        sent_id_ls+=sent_id
        sent_id+=' '
        tags_ls+=' '
        tokens_ls+=' '
    zippedList =  list(zip(sent_id_ls, tokens_ls, tags_ls))
    dfObj = pd.DataFrame(zippedList, columns = ['sentence_id' , 'token', 'tag']) 
    return dfObj

def convert_data_to_connl_with_splitting_data(df,percentage=0.6):    
    
    df=df.sort_values(by='sentence')
    new_data=[]
    idx=df.index
    line=df.loc[idx[0]]
    row=[line.sentence,[(line.start,line.end),line.one_mention]]
    for i in idx[1:len(idx)]:
        line2=df.loc[i]
        if line.sentence==line2.sentence:
            row.append([(line2.start,line2.end),line2.one_mention])
        else : 
            line=df.loc[i]
            new_data.append(row)
            row=[line.sentence,[(line.start,line.end),line.one_mention]]
    new_data.append(row)    
    
    X_train, X_test = train_test_split(new_data, test_size=percentage)
#     X_test, X_dev = train_test_split(X_test, test_size=0.33)
    train=convert(X_train)
    test=convert(X_test)
#     dev=convert(X_dev)
    
    
    return train,test


parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str, help='the data to process', required=True)
parser.add_argument('--percentage', help='the percentage to test Bert model and train the classifier', required=True)
args=parser.parse_args()


print('start preprocessing...')
start_time = tme.time()

df_ontonotes_train=parse_data(args.data)
df_ontonotes_train=clear_mention(df_ontonotes_train)

cnLL_df_ontonotes_train_,cnLL_df_ontonotes_test_=convert_data_to_connl_with_splitting_data(df_ontonotes_train,args.percentage)
cnLL_df_ontonotes_train_[['token','tag']].to_csv('train.txt', header=False,index=None,sep='\t')
cnLL_df_ontonotes_test_[['token','tag']].to_csv('test.txt', header=False,index=None,sep='\t')

print('end')
print("--- %s seconds ---" % (tme.time() - start_time))
