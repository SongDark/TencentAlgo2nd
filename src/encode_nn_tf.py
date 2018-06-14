# coding:utf-8
import pandas as pd
import numpy as np
import gc, os, time
from scipy import sparse

static_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','aid',
                 'advertiserId','campaignId', 'creativeId', 'adCategoryId', 
                 'productId', 'productType', 'creativeSize'] # 15
dynamic_feature=['kw1','kw2','kw3',
                'topic1','topic2','topic3','os','ct','marriageStatus',
                'interest1', 'interest2','interest3','interest4','interest5'] # 14
dynamic_max_len = 30


'''prepare for vocabulary'''
print 'preparing for vocabulary'
# static fields vocabulary
thresholds = {}
for col in dynamic_feature:
    thresholds[col] = 10
thresholds['kw1'] = 50

static_vocabulary = {}
counts_static = pd.read_csv('../data/counts/static_counts.csv')
counts_static.sort_values(by=['field', 'value'], inplace=True)
counts_static.reset_index(inplace=True)
counts_static['rank'] = counts_static.index + 1
for (f, v, r) in counts_static[['field','value','rank']].values:
    static_vocabulary[f+'='+str(v)] = r

# dynamic fields vocabulary
dynamic_vocabulary = {}
counts_dynamic = pd.read_csv('../data/counts/dynamic_counts.csv')
counts_dynamic['ref'] = map(lambda field,total,value: field+'='+str(value) if total>thresholds[field] else field+'=less', 
                            counts_dynamic['field'], counts_dynamic['total'], counts_dynamic['value'])
counts_dynamic.sort_values(by=['field', 'ref', 'value'], inplace=True)
ref = counts_dynamic[['ref']].drop_duplicates().reset_index()
ref['rank'] = ref.index + 1
counts_dynamic = pd.merge(counts_dynamic, ref[['ref','rank']], 'left', ['ref'])
for (f, v, r) in counts_dynamic[['field','value','rank']].values:
    dynamic_vocabulary[f+'='+str(v)] = r
print max(static_vocabulary.values()), len(np.unique(static_vocabulary.values()))
print max(dynamic_vocabulary.values()), len(np.unique(dynamic_vocabulary.values()))
for x in dynamic_vocabulary.keys():
    if x=='kw1':
        print x

'''Encode'''
def to_sparse(lst):
    rows, cols, data = [], [], []
    for i, line in enumerate(lst):
        L = min([len(line), dynamic_max_len])
        rows.extend([i]*L)
        cols.extend(range(L))
        data.extend(line[:dynamic_max_len])
    data = np.array(data, dtype=int)
    return sparse.coo_matrix((data, (rows, cols)), shape=(len(lst), dynamic_max_len))

def prepare(df, flag):
    print 'preparing {0}'.format(flag)
    to_save = '../data/encoded_tf/{0}_{1}'
    df[static_feature].to_csv(to_save.format(flag, 'static_index.csv'), index=False)

    for col in static_feature:
        del df[col]
        gc.collect()

    dynamic_lengths = np.hstack([df[col].apply(lambda vec:min(len(vec), dynamic_max_len)).values[:,None] for col in dynamic_feature])
    pd.DataFrame(dynamic_lengths, columns=dynamic_feature).to_csv(to_save.format(flag, 'dynamic_lengths.csv'), index=False)
    del dynamic_lengths
    gc.collect()
    
    dynamic_index = to_sparse(df[dynamic_feature[0]].values)
    del df[dynamic_feature[0]]
    gc.collect()
    for col in dynamic_feature[1:]:
        dynamic_index = sparse.hstack([dynamic_index, to_sparse(df[col].values)])
        del df[col]
        gc.collect()

    sparse.save_npz(to_save.format(flag, 'dynamic_index.npz'), dynamic_index)
    del dynamic_index, df
    gc.collect()

def encode(flag):
    print 'encoding {0}'.format(flag)
    df = pd.read_csv('../data/origin/{0}.csv'.format(flag))
    for col in static_feature:
        df[col] = map(lambda v : static_vocabulary[col+'='+str(int(float(v)))], df[col])
    for col in dynamic_feature:
        df[col] = map(lambda vec : [dynamic_vocabulary[col+'='+str(int(float(v)))] for v in vec.split(' ')], df[col])
    prepare(df, flag)
    del df
    gc.collect()

print 'Encoding...'
start = time.time()
if not os.path.exists('../data/encoded_tf/'):
    os.makedirs('../data/encoded_tf/')
for flag in ['train_sample_1_merged', 'train_sample_2_merged', 'train_sample_3_merged', 'train_sample_4_merged', 'test_merged']:
    encode(flag)
print 'finished, cost', time.time()-start, 'sec'