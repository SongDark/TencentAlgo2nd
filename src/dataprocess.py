# coding:utf-8

import pandas as pd
import os, time
from sklearn.model_selection import train_test_split

'''转化.data到.csv，占用内存较小'''
print 'converting userFeature.data into userFeature.csv...'
start = time.time()
headers = ['uid','age','gender','marriageStatus','education','consumptionAbility','LBS',
           'interest1','interest2','interest3','interest4','interest5',
           'kw1','kw2','kw3','topic1','topic2','topic3','appIdInstall','appIdAction','ct','os','carrier','house']
with open('../data/userFeature.data', 'rb') as fin, open('../data/userFeature.csv', 'wb') as fout:
    fout.write(",".join(headers)+"\n") # 写入header
    for i,line in enumerate(fin):
        line = line[:-1].split('|') # 去掉\n再split
        userFeature_dict = {}
        for each in line:
            each_list = each.split(' ')
            userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
        line_to_write = []
        for key in headers:
            if userFeature_dict.has_key(key):
                line_to_write.append(userFeature_dict[key])
            else:
                line_to_write.append("")
        fout.write(",".join(line_to_write)+"\n")
        if i%200000==0:
            print 'line', i
print 'finished, cost', time.time() - start, 'sec' 


'''划分训练集和测试集'''
print 'dividing train.csv into 4 sub-parts...'
start = time.time()
train = pd.read_csv('../data/train.csv')
train_dup = train.drop_duplicates(['aid'])
aid_list = list(train_dup['aid'])
num = 1
for i in aid_list:
    temp = train[train.aid==i]
    training_data = temp[['aid', 'uid', 'label']]
    label = temp['label']
    train_1_x, train_2_x, train_1_y, train_2_y = train_test_split(training_data, label, test_size=0.5, random_state=19931028)
    train_1_1_x, train_1_2_x, train_1_1_y, train_1_2_y = train_test_split(train_1_x, train_1_y, test_size=0.5, random_state=19931028)
    train_2_1_x, train_2_2_x, train_2_1_y, train_2_2_y = train_test_split(train_2_x, train_2_y, test_size=0.5, random_state=19931028)
    if num==1:
        train_sample_1 = train_1_1_x
        train_sample_2 = train_1_2_x
        train_sample_3 = train_2_1_x
        train_sample_4 = train_2_2_x
    else:
        train_sample_1 = pd.concat([train_sample_1, train_1_1_x], axis=0)
        train_sample_2 = pd.concat([train_sample_2, train_1_2_x], axis=0)
        train_sample_3 = pd.concat([train_sample_3, train_2_1_x], axis=0)
        train_sample_4 = pd.concat([train_sample_4, train_2_2_x], axis=0)
    num+=1
    del training_data, label, train_1_x, train_2_x, train_1_y, train_2_y
    del train_1_1_x, train_1_2_x, train_1_1_y, train_1_2_y
    del train_2_1_x, train_2_2_x, train_2_1_y, train_2_2_y

train_sample_1.sample(frac=1,random_state=0).reset_index(drop=True).to_csv('../data/train_sample_1.csv', index=False, index_label=False)
train_sample_2.sample(frac=1,random_state=0).reset_index(drop=True).to_csv('../data/train_sample_2.csv', index=False, index_label=False)
train_sample_3.sample(frac=1,random_state=0).reset_index(drop=True).to_csv('../data/train_sample_3.csv', index=False, index_label=False)
train_sample_4.sample(frac=1,random_state=0).reset_index(drop=True).to_csv('../data/train_sample_4.csv', index=False, index_label=False)
del train_sample_1, train_sample_2, train_sample_3, train_sample_4
print 'finished, cost', time.time() - start, 'sec'


'''合并'''
print 'merging'
start = time.time()
if not os.path.exists('../data/origin/'):
    os.makedirs('../data/origin/')

adFeature = pd.read_csv('../data/adFeature.csv')
userFeature = pd.read_csv('../data/userFeature.csv')

for i in range(1,5):
    print 'merging train_sample_{0}'.format(i)
    train = pd.read_csv('../data/train_sample_{0}.csv'.format(i))
    train.loc[train['label']==-1,'label']=0
    train = pd.merge(train, adFeature, 'left', ['aid'])
    train = pd.merge(train, userFeature, 'left', ['uid'])
    train = train.fillna('-1')
    train.to_csv('../data/origin/train_sample_{0}_merged.csv'.format(i), index=False)
    del train
print 'merging test'
test = pd.read_csv('../data/test1.csv')
test['label'] = -1
test = pd.merge(test, adFeature, 'left', ['aid'])
test = pd.merge(test, userFeature, 'left', ['uid'])
test = test.fillna('-1')
test.to_csv('../data/origin/test_merged.csv', index=False)
del test
print 'finished, cost', time.time() - start, 'sec'