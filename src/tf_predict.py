import sys
sys.path.append("../models/")
import os
from TF.deepffm import DeepFFM
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score


def load_numeric_features(flag):
    flag = flag[:-7]
    n = None
    # uid_count, 1
    uid_count = pd.read_csv('../data/statics/uid_count/{0}_uid_count.csv'.format(flag), nrows=n)['uid_count'].values[:,None]
    # aid_col_ratio, 7
    aid_col_ratio = pd.read_csv('../data/statics/aid_col_ratio/{0}_aid_col_ratio.csv'.format(flag), nrows=n).values
    # aid_col_convert_ratio, 15
    aid_col_convert_ratio = pd.read_csv('../data/statics/aid_col_convert_ratio/{0}_aid_col_convert_ratio.csv'.format(flag), nrows=n).values
    return np.hstack([uid_count, aid_col_ratio, aid_col_convert_ratio])

field_sizes = [15, 14, 23]
total_feature_sizes = [1550, 108000, 23]
dropout_fm = [1., 1.]
dropout_deep = [1., 1., 1.]
dynamic_max_len = 30
learning_rate = 0.001
weight_decay = 0.
batch_size = 2048
epoch = 1
latent = 10

flag = 'test_merged'
file_path = '../data/encoded_tf/{0}_{1}'
static_index = pd.read_csv(file_path.format(flag,'static_index.csv')).values
dynamic_index = sparse.load_npz(file_path.format(flag,'dynamic_index.npz')).toarray()
dynamic_lengths = pd.read_csv(file_path.format(flag,'dynamic_lengths.csv')).values
numeric_value = load_numeric_features(flag)
y = pd.read_csv('../data/origin/{0}.csv'.format(flag), usecols=['label']).values
print 'data loaded'

if not os.path.exists('../save/TF/prediction/'):
    os.makedirs('../save/TF/prediction/')

preds = []
for i in range(1, 5):
    init_model = '../save/TF/part{0}/model.ckpt'.format(i)
    print init_model
    dfm = DeepFFM(field_sizes=field_sizes, total_feature_sizes=total_feature_sizes, latent=latent,
            dynamic_max_len=dynamic_max_len, learning_rate=learning_rate, dropout_fm=dropout_fm, dropout_deep=dropout_deep,
            reg_l2=weight_decay, epoch=epoch, batch_size=batch_size, init_model=init_model )
    print 'model built'
    pred = dfm.predict(static_index, dynamic_index, dynamic_lengths, numeric_value, y)
    np.save('../save/TF/prediction/pred{0}.npy'.format(i), np.array(pred))
    preds.append(np.array(pred))

index = pd.read_csv('../data/test1.csv')
index['score'] = (preds[0] + preds[1] + preds[2] + preds[3]) / 4.0
index['score'] = index['score'].apply(lambda x: float('%.6f' % x))
index.to_csv('../submission.csv', index=False)

    


