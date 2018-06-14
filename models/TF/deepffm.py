# coding:utf-8
import time
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class DeepFFM(BaseEstimator, TransformerMixin):
    def __init__(self, field_sizes, total_feature_sizes,
                 dynamic_max_len=30, latent=10, dropout_fm=[1., 1.],
                 deep_layers=[128,64,32], dropout_deep=[1., 1., 1.],
                 deep_activation=tf.nn.relu,
                 epoch=10, batch_size=4096,
                 learning_rate=0.001, optimizer_type='adam',
                 batch_norm=1, batch_norm_decay=0.995,
                 verbose=True, random_state=931028,
                 loss_type='logloss', eval_metric=roc_auc_score,
                 init_model=None, save_model=None,
                 reg_l2=0., greater_is_better=True):

        self.field_sizes = field_sizes # [static_size, dynamic_size, numeric_size]
        self.total_field_size = field_sizes[0] + field_sizes[1] + field_sizes[2]
        self.total_feature_sizes = total_feature_sizes # [static_feat_size, dynamic_feat_size, numeric_feat_size]
        self.latent = latent
        self.dynamic_max_len = dynamic_max_len

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_activation = deep_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.verbose = verbose
        self.random_state = random_state

        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.reg_l2 = reg_l2
        self.greater_is_better = greater_is_better

        self.init_model = init_model
        self.save_model = save_model

        self._init_graph()
        print 'DeepFFM with numeric'
    
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            tf.set_random_seed(self.random_state) # set seed

            self.static_index = tf.placeholder(tf.int32, [None, None], 'static_index') # None * static_feature_size
            
            self.dynamic_index = tf.placeholder(tf.int32, [None, None], 'dynamic_index') # None * [dynamic_feature_size * max_len]
            self.dynamic_lengths = tf.placeholder(tf.int32, [None, None], 'dynamic_lengths') # None * dynamic_feature_size

            self.numeric_value = tf.placeholder(tf.float32, [None, None], 'numeric_value') # None * numeric_feature_size

            self.label = tf.placeholder(tf.float32, [None, 1], 'label')

            self.dropout_keep_fm = tf.placeholder(tf.float32, [None], 'dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32, [None], 'dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool,  name='train_phase')

            '''initialize weights'''
            self.weights = self._init_weights()

            '''lr part'''
            #  None * static_feature_size * 1
            self.static_embs_lr = tf.nn.embedding_lookup(self.weights['static_embs_lr'], self.static_index) 
            self.static_embs_lr = tf.reshape(self.static_embs_lr, [-1, self.field_sizes[0]])  # None * static_feature_size

            # None * [dynamic_feature_size * max_len] * 1
            self.dynamic_embs_lr = tf.nn.embedding_lookup(self.weights['dynamic_embs_lr'], self.dynamic_index) 
            # None * dynamic_feature_size * max_len
            self.dynamic_embs_lr = tf.reshape(self.dynamic_embs_lr, [-1, self.field_sizes[1], self.dynamic_max_len]) 
            self.dynamic_embs_lr = tf.reduce_sum(self.dynamic_embs_lr, axis=2) # None * dynamic_feature_size
            self.dynamic_embs_lr = tf.div(self.dynamic_embs_lr, tf.to_float(self.dynamic_lengths)) # None * dynamic_feature_size

            self.numeric_embs_lr = tf.matmul(self.numeric_value, self.weights['numeric_embs_lr']) # None * 1

            '''ffm part'''
            # None * static_feature_size * [F * K]
            self.static_embs_ffm = tf.nn.embedding_lookup(self.weights['static_embs_ffm'], self.static_index)
            
            # None * [dynamic_feature_size * max_len] * [F * K]
            self.dynamic_embs_ffm = tf.nn.embedding_lookup(self.weights['dynamic_embs_ffm'], self.dynamic_index)
            # None * dynamic_feature_size * max_len * [F * K]
            self.dynamic_embs_ffm = tf.reshape(self.dynamic_embs_ffm, 
                [-1, self.field_sizes[1], self.dynamic_max_len, self.latent * self.total_field_size])
            # None * numeric_feature_size * [F * K]
            self.numeric_embs_ffm = tf.matmul(self.numeric_value, self.weights['numeric_embs_ffm'])
            self.numeric_embs_ffm = tf.reshape(self.numeric_embs_ffm, [-1, self.field_sizes[2], self.latent * self.total_field_size])
             
            # [None * dynamic_feature_size] * max_len
            self.ffm_mask = tf.sequence_mask(tf.reshape(self.dynamic_lengths, [-1]), maxlen=self.dynamic_max_len)
            self.ffm_mask = tf.expand_dims(self.ffm_mask, -1) # [None * dynamic_feature_size] * max_len * 1
            # [None * dynamic_feature_size] * max_len * [F * K]
            self.ffm_mask = tf.concat([self.ffm_mask for _ in range(self.latent * self.total_field_size)], axis=-1) 
            # [None * dynamic_feature_size] * max_len * [F * K]
            self.dynamic_embs_ffm = tf.reshape(self.dynamic_embs_ffm, 
                [-1, self.dynamic_max_len, self.latent * self.total_field_size])
            self.dynamic_embs_ffm = tf.multiply(self.dynamic_embs_ffm, tf.to_float(self.ffm_mask))
            self.dynamic_embs_ffm = tf.reshape(tf.reduce_sum(self.dynamic_embs_ffm, axis=1), 
                [-1, self.field_sizes[1], self.latent * self.total_field_size]) # None * dynamic_feature_size * [F * K]
            
            # None * dynamic_feature_size * [F * K]
            self.padding_lengths = tf.concat([tf.expand_dims(self.dynamic_lengths, -1) for _ in range(self.latent * self.total_field_size)], axis=-1)
            self.dynamic_embs_ffm = tf.div(self.dynamic_embs_ffm, tf.to_float(self.padding_lengths))

            # None * F * F * K
            self.embs_ffm_col = tf.reshape(tf.concat([self.static_embs_ffm, self.dynamic_embs_ffm, self.numeric_embs_ffm], axis=1), 
                                    [-1, self.total_field_size, self.total_field_size, self.latent])
            # None * F * F * K
            self.embs_ffm_row = tf.transpose(self.embs_ffm_col, [0,2,1,3]) 
            self.embs_ffm = tf.multiply(self.embs_ffm_col, self.embs_ffm_row)

            self.ones = tf.ones_like(self.embs_ffm)
            self.op = tf.linalg.LinearOperatorLowerTriangular(tf.transpose(self.ones, [0,3,1,2])) # None * K * F * F
            self.upper_tri_mask = tf.less(tf.transpose(self.op.to_dense(), [0,2,3,1]), self.ones) # None * F * F * K

            self.embs_ffm = tf.boolean_mask(self.embs_ffm, self.upper_tri_mask)
            self.embs_ffm = tf.reshape(self.embs_ffm, [-1, self.total_field_size*(self.total_field_size-1)//2 * self.latent])


            '''deep part'''
            self.y_deep = self.embs_ffm
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(len(self.deep_layers)):
                self.y_deep = tf.matmul(self.y_deep, self.weights['layer_%d'%i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn='bn_%d'%i)
                self.y_deep = self.deep_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i])
            
            '''deep ffm'''
            self.out = tf.add(tf.matmul(self.y_deep, self.weights['concat_projection']), self.weights['concat_bias'])
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.static_embs_lr, axis=1), [-1,1]))
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.dynamic_embs_lr, axis=1), [-1,1]))
            self.out = tf.add(self.out, tf.reshape(self.numeric_embs_lr, [-1,1]))
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.embs_ffm, axis=1), [-1,1]))
            
            '''loss'''
            if self.loss_type=='logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            '''regularization'''
            if self.reg_l2 > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.reg_l2)(self.weights['concat_projection'])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.reg_l2)(self.weights['layer_%d'%i])
            
            '''optimizer'''
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # init model 
            if self.init_model!=None:
                self.saver.restore(self.sess, self.init_model)

            # number of params
            total_params = 0
            for v in self.weights.values():
                shape = v.get_shape()
                num = 1.0
                for dim in shape:
                    num *= dim.value
                total_params += num
            if self.verbose > 0:
                print 'params', total_params
    
    def _init_session(self):
        config = tf.ConfigProto(device_count={'gpu':0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _init_weights(self):
        weights = {}

        # LR
        weights['static_embs_lr'] = tf.Variable(tf.random_normal((self.total_feature_sizes[0], 1), mean=0.0, stddev=1e-4), name='static_embs_lr')
        weights['dynamic_embs_lr'] = tf.Variable(tf.random_normal((self.total_feature_sizes[1], 1), mean=0.0, stddev=1e-4), name='dynamic_embs_lr')
        weights['numeric_embs_lr'] = tf.Variable(tf.random_normal((self.field_sizes[2], 1), mean=0.0, stddev=1e-4), name='numeric_embs_lr')

        # ffm
        # static_feature_size * [F*K]
        weights['static_embs_ffm'] = tf.Variable(
            tf.random_normal((self.total_feature_sizes[0], self.latent * self.total_field_size), mean=0.0, stddev=1e-4), name='static_embs_ffm')
        # dynamic_feature_size * [F*K]
        weights['dynamic_embs_ffm'] = tf.Variable(
            tf.random_normal((self.total_feature_sizes[1], self.latent * self.total_field_size), mean=0.0, stddev=1e-4), name='dynamic_embs_ffm')
        # numeric_feature_size * [F*K]
        weights['numeric_embs_ffm'] = tf.Variable(
            tf.random_normal((self.field_sizes[2], self.field_sizes[2] * self.latent * self.total_field_size), mean=0.0, stddev=1e-4), name='numeric_embs_ffm')


        # deep
        num_of_layers = len(self.deep_layers)
        input_size = self.total_field_size * (self.total_field_size - 1)//2 * self.latent # F*(F-1)/2*K
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32, name='layer_0')
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32, name='bias_0')
        
        for i in range(1, num_of_layers):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights['layer_%d' % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])), dtype=np.float32, name='layer_%d' % i)
            weights['bias_%d' % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])), dtype=np.float32, name='bias_%d' % i)
        
        input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(-3.5), dtype=np.float32)

        return weights
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda : bn_train, lambda : bn_inference)
        return z
    
    def get_batch(self, static_index, dynamic_index, dynamic_lengths, numeric_value, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end<len(y) else len(y)
        return static_index[start:end], dynamic_index[start:end], dynamic_lengths[start:end], numeric_value[start:end], y[start:end]
    
    def shuffle_in_unison_scary(self, a, b, c, d, e):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)

    def fit_on_batch(self, static_index, dynamic_index, dynamic_lengths, numeric_value, y):
        # print 'fit on batch', static_index.shape, dynamic_index.shape, dynamic_lengths.shape, y.shape
        feed_dict = {self.static_index:static_index,
                     self.dynamic_index:dynamic_index,
                     self.dynamic_lengths:dynamic_lengths,
                     self.numeric_value:numeric_value,
                     self.label:y,
                     self.dropout_keep_fm:self.dropout_fm,
                     self.dropout_keep_deep:self.dropout_deep,
                     self.train_phase:True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    
    def fit(self, 
            train_static_index, train_dynamic_index, train_dynamic_lengths, train_numeric_value, train_y,
            valid_static_index=None, valid_dynamic_index=None, valid_dynamic_lengths=None, valid_numeric_value=None, valid_y=None):
        print train_static_index.shape, train_dynamic_index.shape, train_dynamic_lengths.shape, train_numeric_value.shape, train_y.shape
        has_valid = valid_static_index is not None
        if has_valid:
            print valid_static_index.shape, valid_dynamic_index.shape, valid_dynamic_lengths.shape, valid_numeric_value.shape, valid_y.shape

        for epoch in range(self.epoch):
            total_loss = 0.0
            total_size = 0.0
            batch_begin_time = time.time()
            t1 = time.time()
            self.shuffle_in_unison_scary(train_static_index, train_dynamic_index, train_dynamic_lengths, train_numeric_value, train_y)
            total_batch = int(len(train_y) / self.batch_size)
            for i in range(total_batch):
                offset = i*self.batch_size
                end = (i+1)*self.batch_size
                end = end if end<len(train_y) else len(train_y)
                static_index_batch, dynamic_index_batch, dynamic_lengths_batch, numeric_value_batch, y_batch = \
                    self.get_batch(train_static_index, train_dynamic_index, train_dynamic_lengths, train_numeric_value, train_y, self.batch_size, i)
                batch_loss = self.fit_on_batch(static_index_batch, dynamic_index_batch, dynamic_lengths_batch, numeric_value_batch, y_batch)
                total_loss += batch_loss * (end - offset)
                total_size += end - offset
                if i%100==99:
                    print '[%d, %5d] loss : %.6f time: %.1f s' % (epoch+1, i+1, total_loss/total_size, time.time()-batch_begin_time)
                    total_loss = 0.0
                    total_size = 0.0
                    batch_begin_time = time.time()
            
            print 'train',
            train_result = self.evaluate(train_static_index, train_dynamic_index, train_dynamic_lengths, train_numeric_value, train_y)

            if has_valid:
                print 'valid',
                valid_result = self.evaluate(valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_numeric_value, valid_y)
            
            print
            if self.verbose > 0:
                if has_valid:
                    print '[%d] train_result=%.4f, valid_result=%.4f [%.1f s]' % (epoch+1, train_result, valid_result, time.time()-t1)
                else:
                    print '[%d] train_result=%.4f [%.1f s]' % (epoch+1, train_result, time.time()-t1)

        # save the model
        if self.save_model!=None:
            self.saver.save(self.sess, self.save_model) 
            print 'saved to', self.save_model      

    
    def predict(self, static_index, dynamic_index, dynamic_lengths, numeric_value, y=[]):
        if len(y)==0:
            dummy_y = [1] * len(static_index)
        else:
            dummy_y = y
        batch_index = 0
        batch_size = 2048
        static_index_batch, dynamic_index_batch, dynamic_lengths_batch, numeric_value_batch, y_batch = \
            self.get_batch(static_index, dynamic_index, dynamic_lengths, numeric_value, dummy_y, batch_size, 0)
        y_pred = None
        total_loss = 0.0
        total_size = 0.0
        while len(static_index_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.static_index:static_index_batch,
                         self.dynamic_index:dynamic_index_batch,
                         self.dynamic_lengths:dynamic_lengths_batch,
                         self.numeric_value:numeric_value_batch,
                         self.label:y_batch,
                         self.dropout_keep_fm:[1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep:[1.0] * len(self.dropout_deep),
                         self.train_phase:False}
            batch_out, batch_loss = self.sess.run((self.out, self.loss), feed_dict=feed_dict)
            total_loss += batch_loss * num_batch
            total_size += num_batch
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate([y_pred, np.reshape(batch_out, (num_batch,))])
            
            batch_index += 1
            static_index_batch, dynamic_index_batch, dynamic_lengths_batch, numeric_value_batch, y_batch = \
                self.get_batch(static_index, dynamic_index, dynamic_lengths, numeric_value, dummy_y, batch_size, batch_index)
        print 'logloss = %.6f' % (total_loss / total_size),
        return y_pred

    def evaluate(self, static_index, dynamic_index, dynamic_lengths, numeric_value, y):
        
        y_pred = self.predict(static_index, dynamic_index, dynamic_lengths, numeric_value, y)

        res = self.eval_metric(y, y_pred)

        return res
        