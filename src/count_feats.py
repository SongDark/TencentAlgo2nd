# coding:utf-8

'''
    用于统计各field不同取值的出现次数，保存到count.csv中
'''

import csv, os, time
import collections
import numpy as np
import pandas as pd

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','aid','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType', 'creativeSize'] # 15
       
vector_feature=['kw1','kw2','kw3','topic1','topic2','topic3','os','ct','marriageStatus', 
                'interest1', 'interest2','interest3','interest4','interest5'] # 14

def count_static_feats(csv_path, retDict):
    print csv_path
    for i, row in enumerate(csv.DictReader(open(csv_path)), start=1):
        for field in row:
            # categorical 特征
            if field in one_hot_feature:
                value = str(int(float(row[field])))
                retDict[field+','+value][0] += 1
        # 显示进度
        if i%250000==0:
            print i

def count_dynamic_feats(csv_path, retDict):
    print csv_path
    for i, row in enumerate(csv.DictReader(open(csv_path)), start=1):
        for field in row:
            # 向量型特征，需要先split
            if field in vector_feature:
                values = row[field].split(' ')
                for value in values:
                    value = str(int(float(value)))
                    retDict[field+','+value][0] += 1
        # 显示进度
        if i%250000==0:
            print i

start = time.time()
if not os.path.exists('../data/counts/'):
    os.makedirs('../data/counts/')
static_counts = collections.defaultdict(lambda : [0])
count_static_feats('../data/origin/train_sample_1_merged.csv', static_counts)
count_static_feats('../data/origin/train_sample_2_merged.csv', static_counts)
count_static_feats('../data/origin/train_sample_3_merged.csv', static_counts)
count_static_feats('../data/origin/train_sample_4_merged.csv', static_counts)
count_static_feats('../data/origin/test_merged.csv', static_counts)
# 保存
with open('../data/counts/static_counts.csv','wb') as f:
    f.write("field,value,total\n")
    for key, [total] in sorted(static_counts.items(), key=lambda x:x[1][0], reverse=True):
        f.write(','.join([str(key), str(total)]))
        f.write('\n')
del static_counts

dynamic_counts = collections.defaultdict(lambda : [0])
count_dynamic_feats('../data/origin/train_sample_1_merged.csv', dynamic_counts)
count_dynamic_feats('../data/origin/train_sample_2_merged.csv', dynamic_counts)
count_dynamic_feats('../data/origin/train_sample_3_merged.csv', dynamic_counts)
count_dynamic_feats('../data/origin/train_sample_4_merged.csv', dynamic_counts)
count_dynamic_feats('../data/origin/test_merged.csv', dynamic_counts)
# 保存
with open('../data/counts/dynamic_counts.csv','wb') as f:
    f.write("field,value,total\n")
    for key, [total] in sorted(dynamic_counts.items(), key=lambda x:x[1][0], reverse=True):
        f.write(','.join([str(key), str(total)]))
        f.write('\n')
del dynamic_counts
print 'finished, cost', time.time()-start, 'sec'