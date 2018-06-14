# coding:utf-8
import pandas as pd
import os, time
import collections
import csv
import numpy as np

if not os.path.exists('../data/statics/'):
    os.makedirs('../data/statics/')


'''uid 出现次数'''
print 'static feature : uid_count'
start = time.time()
train_1 = pd.read_csv('../data/origin/train_sample_1_merged.csv', usecols=['uid'])
train_2 = pd.read_csv('../data/origin/train_sample_2_merged.csv', usecols=['uid'])
train_3 = pd.read_csv('../data/origin/train_sample_3_merged.csv', usecols=['uid'])
train_4 = pd.read_csv('../data/origin/train_sample_4_merged.csv', usecols=['uid'])
test = pd.read_csv('../data/origin/test_merged.csv', usecols=['uid'])
uid = pd.concat([train_1, train_2, train_3, train_4, test])
uid_size = uid.groupby(['uid']).size().reset_index().rename(columns={0:'uid_count'})

# 归一化
min_c = min(uid_size['uid_count'].values)
max_c = max(uid_size['uid_count'].values)
delta = float(max_c - min_c)
uid_size['uid_count'] = uid_size['uid_count'].apply(lambda x:(x-min_c)/delta)
# uid_size.sort_values(by=['uid_count'], ascending=False).to_csv('/media/statics/uid_count/uid_size.csv', index=False)

if not os.path.exists('../data/statics/uid_count/'):
    os.makedirs('../data/statics/uid_count/')
train_1 = pd.merge(train_1, uid_size, 'left', ['uid'])
train_1.to_csv('../data/statics/uid_count/train_sample_1_uid_count.csv', index=False)
train_2 = pd.merge(train_2, uid_size, 'left', ['uid'])
train_2.to_csv('../data/statics/uid_count/train_sample_2_uid_count.csv', index=False)
train_3 = pd.merge(train_3, uid_size, 'left', ['uid'])
train_3.to_csv('../data/statics/uid_count/train_sample_3_uid_count.csv', index=False)
train_4 = pd.merge(train_4, uid_size, 'left', ['uid'])
train_4.to_csv('../data/statics/uid_count/train_sample_4_uid_count.csv', index=False)

test = pd.merge(test, uid_size, 'left', ['uid'])
test.to_csv('../data/statics/uid_count/test_uid_count.csv', index=False)
del train_1, train_2, train_3, train_4, test, uid, uid_size
print 'finished, cost', time.time()-start, 'sec'



'''分布统计特征'''
start = time.time()
print 'static feature : aid_col_count'
static_feature=['LBS','age','carrier','consumptionAbility','education','gender','house']

if not os.path.exists('../data/statics/aid_col_ratio/'):
    os.makedirs('../data/statics/aid_col_ratio/')

data = []
for i in range(4):
    data.append(pd.read_csv('../data/train_sample_{0}.csv'.format(i+1), usecols=['aid']))
data.append(pd.read_csv('../data/test1.csv', usecols=['aid']))
data = pd.concat(data)
aid_count = data.groupby(['aid']).size().reset_index().rename(columns={0:'aid_count'})
del data

for col in static_feature:
    des_col = 'aid_{0}_ratio'.format(col)
    print col, des_col
    data = []
    for i in range(4):
        data.append(pd.read_csv('../data/origin/train_sample_{0}_merged.csv'.format(i+1), usecols=['aid', col]))
        data[-1]['flag'] = i+1
    data.append(pd.read_csv('../data/origin/test_merged.csv', usecols=['aid', col]))
    data[-1]['flag'] = -1
    data = pd.concat(data)

    aid_col_count = data.groupby(['aid', col]).size().reset_index().rename(columns={0:'aid_{0}_count'.format(col)})
    aid_col_count = pd.merge(aid_col_count, aid_count, 'left', on=['aid'])
    aid_col_count[des_col] = aid_col_count['aid_{0}_count'.format(col)] / aid_col_count['aid_count']

    data = pd.merge(data, aid_col_count[['aid',col,des_col]], 'left', on=['aid', col])

    for i in range(4):
        data[data.flag==i+1][[des_col]].to_csv('../data/statics/aid_col_ratio/train_sample_{0}_{1}.csv'.format(i+1, des_col), index=False)
    data[data.flag==-1][[des_col]].to_csv('../data/statics/aid_col_ratio/test_{0}.csv'.format(des_col), index=False)
    del data

# 归一化并拼接成大csv
mins, maxs = {}, {}
for col in static_feature:
    mins[col], maxs[col] = [], []

for flag in ['train_sample_1', 'train_sample_2', 'train_sample_3', 'train_sample_4', 'test']:
    for col in static_feature:
        data = pd.read_csv('../data/statics/aid_col_ratio/{0}_aid_{1}_ratio.csv'.format(flag, col), nrows=None)
        mins[col].append(min(data['aid_'+col+'_ratio'].values))
        maxs[col].append(max(data['aid_'+col+'_ratio'].values))
        # print flag, col, min(data['aid_'+col+'_ratio'].values), max(data['aid_'+col+'_ratio'].values)
        del data
for col in static_feature:
    mins[col] = min(mins[col])
    maxs[col] = max(maxs[col])

for flag in ['train_sample_1', 'train_sample_2', 'train_sample_3', 'train_sample_4', 'test']:
    print flag
    col = static_feature[0]
    data = pd.read_csv('../data/statics/aid_col_ratio/{0}_aid_{1}_ratio.csv'.format(flag, col)) 
    data['aid_'+col+'_ratio'] = data['aid_'+col+'_ratio'].apply(lambda x:(x-mins[col])/(maxs[col]-mins[col])).copy()

    for col in static_feature[1:]:
        tmp = pd.read_csv('../data/statics/aid_col_ratio/{0}_aid_{1}_ratio.csv'.format(flag, col))
        data['aid_'+col+'_ratio'] = tmp['aid_'+col+'_ratio'].apply(lambda x:(x-mins[col])/(maxs[col]-mins[col])).copy()
        del tmp
    data.to_csv('../data/statics/aid_col_ratio/{0}_aid_col_ratio.csv'.format(flag), index=False)

print 'finished, cost', time.time() - start, 'sec'



'''转化率特征'''
print 'static : aid_col_convert_ratio'
combine_feature = ['interest1', 'interest2','interest3','interest4','interest5'] # 5

def count_combine_feats(csv_path, retDict):
    print csv_path
    for i, row in enumerate(csv.DictReader(open(csv_path)), start=1):
        aid = str(int(float(row['aid'])))
        for field in row:
            if field in combine_feature:
                values = row[field].split(' ')
                for value in values:
                    value = str(int(float(value)))
                    key = 'aid_'+field+','+aid+'_'+value
                    retDict[key][2] += 1
                    if float(row['label']) == 1:
                        retDict[key][0] += 1
                    else:
                        retDict[key][1] += 1
        if i%250000==0:
            print i

if not os.path.exists('../data/statics/aid_col_convert_ratio/'):
    os.makedirs('../data/statics/aid_col_convert_ratio/')

# 统计
print 'counting...'
start = time.time()
counts = collections.defaultdict(lambda : [0,0,0])
count_combine_feats('../data/origin/train_sample_1_merged.csv', counts)
count_combine_feats('../data/origin/train_sample_2_merged.csv', counts)
count_combine_feats('../data/origin/train_sample_3_merged.csv', counts)
count_combine_feats('../data/origin/train_sample_4_merged.csv', counts)

with open('../data/statics/aid_col_convert_ratio/counts.csv','wb') as f:
    f.write("field,value,pos,neg,total,ratio\n")
    for key, [pos, neg, total] in sorted(counts.items(), key=lambda x:x[1][2]):
        f.write(','.join([str(key), str(pos), str(neg), str(total), str(pos/float(total))]))
        f.write('\n')
del counts
counts = pd.read_csv('../data/statics/aid_col_convert_ratio/counts.csv')

# 转化
print '\nconverting...'
aid_interest_convert_ratio_dict = {}
for (f, v, p, n, t, r) in counts.values:
    key = str(f) + '=' + str(v) # aid_interest1=12_123
    aid_interest_convert_ratio_dict[key] = r

counts['aid'] = counts['value'].apply(lambda x:x.split('_')[0])
counts['col'] = counts['field'].apply(lambda x:x.split('_')[1])
aid_avg_ratio = counts.groupby(['aid','col'])['ratio'].mean().reset_index().rename(columns={'ratio':'avg_ratio'})

aid_avg_convert_ratio_dict = {}
for (aid, col, r) in aid_avg_ratio.values:
    aid_avg_convert_ratio_dict[str(aid)+'_'+col] = r
del counts, aid_avg_ratio

def convert_prob(aid, col_vec, col_name):
    col_vec = col_vec.split(' ')
    ret = []
    for value in col_vec:
        key = 'aid_'+col_name+'='+aid+'_'+value
        if aid_interest_convert_ratio_dict.has_key(key):
            ret.append(aid_interest_convert_ratio_dict[key])
        else:
            ret.append(aid_avg_convert_ratio_dict[aid+'_'+col_name])
    return ret

for flag in ['train_sample_1', 'train_sample_2', 'train_sample_3', 'train_sample_4', 'test']:
    print flag
    data = pd.read_csv('../data/origin/{0}_merged.csv'.format(flag), usecols=['aid']+combine_feature, nrows=None)
    data['aid'] = data['aid'].apply(str)

    for col in combine_feature:
        data[col+'_ratios'] = map(lambda aid, col_vec:convert_prob(aid, col_vec, col), data['aid'], data[col])
        data[col+'_convert_ratio_avg'] = data[col+'_ratios'].apply(np.mean)
        data[col+'_convert_ratio_min'] = data[col+'_ratios'].apply(min)
        data[col+'_convert_ratio_max'] = data[col+'_ratios'].apply(max)
        del data[col+'_ratios'], data[col]

    data.drop(['aid'], axis=1).to_csv('../data/statics/aid_col_convert_ratio/{0}_aid_col_convert_ratio.csv'.format(flag), index=False)
    del data


# 归一化
print '\nnormalization...'
n = None
mins = []
maxs = []
for flag in ['train_sample_1', 'train_sample_2', 'train_sample_3', 'train_sample_4', 'test']:
    print flag
    data = pd.read_csv('../data/statics/aid_col_convert_ratio/{0}_aid_col_convert_ratio.csv'.format(flag), nrows=n)
    mins.append(np.min(data.values, axis=0))
    maxs.append(np.max(data.values, axis=0))
    del data

mins = np.mean(np.array(mins), axis=0)
maxs = np.mean(np.array(maxs), axis=0)

for flag in ['train_sample_1', 'train_sample_2', 'train_sample_3', 'train_sample_4', 'test']:
    data = pd.read_csv('../data/statics/aid_col_convert_ratio/{0}_aid_col_convert_ratio.csv'.format(flag), nrows=n)
    cols = list(data.columns)
    current_mins = np.tile(mins, (len(data), 1))
    current_maxs = np.tile(maxs, (len(data), 1))
    data = (data.values - current_mins) / (current_maxs - current_mins)
    data = pd.DataFrame(data, columns=cols)
    data.to_csv('../data/statics/aid_col_convert_ratio/{0}_aid_col_convert_ratio.csv'.format(flag), index=False)
    del data

print time.time() - start