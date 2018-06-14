import sys
sys.path.append("../models/")

from TF.deepffm import DeepFFM
import pandas as pd
import numpy as np
from scipy import sparse
import gc
np.random.seed(931028)

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

for i in range(1,5):
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
    init_model = None
    save_model = '../save/TF/part{0}/model.ckpt'.format(i)

    dfm = DeepFFM(field_sizes=field_sizes, total_feature_sizes=total_feature_sizes, latent=latent,
                dynamic_max_len=dynamic_max_len, learning_rate=learning_rate, dropout_fm=dropout_fm, dropout_deep=dropout_deep,
                reg_l2=weight_decay, epoch=epoch, batch_size=batch_size, init_model=init_model, save_model=save_model)
    print 'model built'

    flag = 'train_sample_{0}_merged'.format(i)
    print flag
    file_path = '../data/encoded_tf/{0}_{1}'
    n = None
    train_static_index = pd.read_csv(file_path.format(flag,'static_index.csv'), nrows=n).values
    train_dynamic_index = sparse.load_npz(file_path.format(flag,'dynamic_index.npz')).toarray()
    train_dynamic_lengths = pd.read_csv(file_path.format(flag,'dynamic_lengths.csv'), nrows=n).values
    train_numeric_value = load_numeric_features(flag)
    train_y = pd.read_csv('../data/origin/{0}.csv'.format(flag), usecols=['label'], nrows=n).values
    print 'train loaded'

    valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_numeric_value, valid_y = \
        train_static_index, train_dynamic_index, train_dynamic_lengths, train_numeric_value, train_y

    # training
    dfm.fit(train_static_index, train_dynamic_index, train_dynamic_lengths, train_numeric_value, train_y,
            valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_numeric_value, valid_y)

    del train_static_index, train_dynamic_index, train_dynamic_lengths, train_numeric_value, train_y
    gc.collect()
