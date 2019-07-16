import tensorflow as tf
import cPickle as pkl
import random
import sys
from srnn import BasicModel
from data_loader import DataLoader as DataLoader
#from data_loader import DataLoader_Mul as DataLoader
from data_loader import CroppedLoader as CroppedLoader
#from data_loader import DataLoader_crop as CroppedLoader
from sklearn.metrics import *
import numpy as np
import time
import os
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"



class rum_cell(tf.contrib.rnn.RNNCell):
    def __init__(self, feature_number, num_units, reuse=False, dtype=tf.float32, name=None, **kwargs):
        self._feature_number = feature_number
        self._num_units = num_units
        self.reuse = reuse
        self.step = 0
        super(rum_cell, self).__init__(
            trainable=True, name=name, dtype=dtype, **kwargs)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        return tf.zeros(shape=[batch_size, self._feature_number, self._num_units], dtype=dtype)

    def attention(self, key, value, query):
        # key, value: [B, T, Dk], query: [B, Dq], mask: [B, T, 1]
        # _, max_len, k_dim = key.get_shape().as_list()
        # query = tf.layers.dense(query, k_dim, activation=None)
        # query = self.prelu(query)
        k = key.get_shape().as_list()[0]
        queries = tf.tile(tf.expand_dims(query, 1), [1, k, 1])  # [B, T, Dk]
        key = queries + key - queries
        inp = tf.concat([queries, key, queries - key, queries * key], axis=-1)
        fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None)  # [B, T, 1]

        # mask = tf.equal(mask, tf.ones_like(mask))  # [B, T, 1]
        # paddings = tf.ones_like(fc3) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.reshape(fc3, [-1, k]))  # [B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2))
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return atten_output_sum, score

    def __call__(self, inputs, state):
        inputs_shape = inputs.shape.as_list()
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
        input_depth = inputs_shape[-1]
        with tf.variable_scope('rum', reuse=(self.step > 0) or self.reuse):
            self._erase_kernel = tf.get_variable(
                "erase/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, self._num_units],
                initializer=tf.random_normal_initializer(stddev=0.1),
            )
            self._erase_bias = tf.get_variable(
                "erase/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=tf.constant_initializer(0, dtype=self.dtype)
            )
            self._add_kernel = tf.get_variable(
                "add/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, self._num_units],
                initializer=tf.random_normal_initializer(stddev=0.1)
            )
            self._add_bias = tf.get_variable(
                "add/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=tf.constant_initializer(0, dtype=self.dtype)
            )
            self._gate_kernel = tf.get_variable(
                "gate/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, self._num_units],
                initializer=tf.random_normal_initializer(stddev=0.1)
            )
            self._gate_bias = tf.get_variable(
                "gate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=tf.constant_initializer(0, dtype=self.dtype)
            )
            self._key = tf.get_variable(
                'key',
                shape=[self._feature_number, input_depth],
                initializer=tf.random_normal_initializer(stddev=0.1)
            )

        # w = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs, axis=1), self._key), axis=-1)
        _, z = self.attention(self._key, state, inputs)
        z = tf.expand_dims(z, -1)
        output = tf.reduce_sum(z * state, axis=1)
        erase = tf.nn.sigmoid(tf.matmul(tf.concat([inputs, output], axis=-1), self._erase_kernel) + self._erase_bias)
        add = tf.nn.sigmoid(
            tf.matmul(tf.concat([inputs, output], axis=1), self._gate_kernel) + self._gate_bias) * tf.nn.tanh(
            tf.matmul(tf.concat([inputs, output], axis=-1), self._add_kernel) + self._add_bias)
        state = state * (1 - z * tf.expand_dims(erase, axis=1)) + z * tf.expand_dims(add, axis=1)
        self.step += 1
        return output, state

    def get_norm(self):
        loss = 0
        for i in range(self._feature_number):
            for j in range(i + 1, self._feature_number):
                loss += tf.reduce_sum(self._key[i, :] * self._key[j, :])
        loss /= tf.nn.l2_loss(self._key)
        return loss

class RUM(BasicModel):
    def __init__(self, path, trainset, testset, item_max_len, user_max_len, item_feature_number,
                 user_feature_number, feature_size, input_dim=1, output_dim=1, learning_rate=0.001, batchsize=256,
                 feature_number=10, hidden_size=32, embedding_size=16, user=True, item=True, emb_initializer=None,
                 beta=0.):
        self.feature_number = feature_number
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.item_max_len = item_max_len
        self.user_max_len = user_max_len
        self.item_feature_number = item_feature_number
        self.user_feature_number = user_feature_number
        self.user = user
        self.item = item
        self.emb_initializer = emb_initializer
        self.feature_size = feature_size
        self.beta = beta
        self.norm = 0
        BasicModel.__init__(self, path, trainset, testset, input_dim, output_dim, learning_rate, batchsize)

    def define_inputs(self):
        self.iin = tf.placeholder(tf.int32, [None, self.item_max_len, self.item_feature_number], name='iin')
        self.ilen = tf.placeholder(tf.int32, [None, ], name='ilen')
        self.uin = tf.placeholder(tf.int32, [None, self.user_max_len, self.user_feature_number], name='uin')
        self.ulen = tf.placeholder(tf.int32, [None, ], name='ulen')
        self.label = tf.placeholder(tf.float32, [None, ], name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

    def build_fc_net(self, inp, keep_prob=0.8):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.elu, name='fc1')
        dp1 = tf.nn.dropout(fc1, keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.elu, name='fc2')
        dp2 = tf.nn.dropout(fc2, keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.prediction = tf.reshape(tf.nn.sigmoid(fc3), [-1, ])

        # loss
        self.log_loss = tf.losses.log_loss(self.label, self.prediction)
        self.cross_entropy = self.log_loss
        for v in tf.trainable_variables():
            self.cross_entropy += self.beta * tf.nn.l2_loss(v)
        self.cross_entropy += 3e-6 * self.norm

        tf.summary.scalar('loss', self.cross_entropy)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = self.optimizer.compute_gradients(self.cross_entropy)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_step = self.optimizer.apply_gradients(capped_gvs)

    def embedding(self):
        if self.emb_initializer is not None:
            emb_mtx = tf.get_variable('emb_mtx', initializer=self.emb_initializer)
        else:
            emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])

        uin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.uin),
                         [-1, self.user_max_len, self.user_feature_number * self.embedding_size])

        iin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.iin),
                         [-1, self.item_max_len, self.item_feature_number * self.embedding_size])

        return uin, iin

    def build_graph(self):
        with tf.variable_scope('embedding'):
            uin, iin = self.embedding()
            index = tf.range(0, self.batch_size * self.user_max_len, self.user_max_len) + self.ulen - 1
            last = tf.gather(tf.reshape(uin, (-1, self.user_feature_number * self.embedding_size)), index)
            uemb = tf.reshape(last[:, :self.embedding_size], [-1, self.embedding_size])
            iemb = tf.reshape(last[:, self.embedding_size:2 * self.embedding_size], [-1, self.embedding_size])
            uin = uin[:, :, self.embedding_size:]
            iin = iin[:, :, :self.embedding_size]

        with tf.variable_scope('user_repre'):
            index = tf.range(0, self.batch_size * self.user_max_len, self.user_max_len) + self.ulen - 1
            cell = rum_cell(self.feature_number, self.hidden_size)
            outputs, state = tf.nn.dynamic_rnn(cell, uin, sequence_length=self.ulen, dtype=tf.float32)
            user_repre = tf.gather(tf.reshape(outputs, (-1, self.hidden_size)), index)
            self.user_norm = cell.get_norm()
            tf.summary.image('user_state', tf.expand_dims(state, -1), 1)
            tf.summary.histogram('user_key', cell._key)

        with tf.variable_scope('item_repre'):
            index = tf.range(0, self.batch_size * self.item_max_len, self.item_max_len) + self.ilen - 1
            cell = rum_cell(self.feature_number, self.hidden_size)
            outputs, state = tf.nn.dynamic_rnn(cell, iin, sequence_length=self.ilen, dtype=tf.float32)
            item_repre = tf.gather(tf.reshape(outputs, (-1, self.hidden_size)), index)
            self.item_norm = cell.get_norm()
            tf.summary.image('item_state', tf.expand_dims(state, -1), 1)
            tf.summary.histogram('item_key', cell._key)

        if self.user:
            if self.item:
                repre = tf.concat([user_repre, uemb, item_repre, iemb], axis=1)
                self.norm = self.user_norm + self.item_norm
            else:
                repre = tf.concat([user_repre, uemb, iemb], axis=1)
                self.norm = self.user_norm
        else:
            repre = tf.concat([uemb, item_repre, iemb], axis=1)
            self.norm = self.item_norm

        with tf.variable_scope('output'):
            self.build_fc_net(repre, self.keep_prob)

        self.merged = tf.summary.merge_all()

    def train(self, epochs=100, p=100):
        step = 0
        flag = True
        losses = []
        for epoch in range(epochs):
            if not flag:
                break
            print('---------------epoch %s---------------' % epoch)
            # random.shuffle(self.trainset)
            start = time.time()
            for _, data in DataLoader(self.trainset, self.batchsize):
                step += 1
                batch_size = len(data[0])
                feed_dict = {
                    self.uin: data[1],
                    self.ulen: data[2],
                    self.iin: data[3],
                    self.ilen: data[4],
                    self.label: data[0],
                    self.batch_size: batch_size,
                    self.keep_prob: 0.5,
                }
                _, merged = self.sess.run(fetches=[self.train_step, self.merged], feed_dict=feed_dict)
                self.summary_writer.add_summary(merged, step)
                if step % p == 0:
                    self.eval(step, self.trainset, 'train')
                    losses.append(self.eval(step, self.testset, 'test'))
                    if len(losses) >= 3 and losses[-1] < losses[-2] < losses[-3]:
                        flag = False
                        break
                    if len(losses) >= 3 and losses[-1] > losses[-2] and losses[-1] > losses[-3]:
                        self.save_model()
            train_time = time.time() - start
            print("Training time: %.4f" % train_time)
        if flag:
            self.save_model()

    def eval(self, step, dataset, prefix):
        labels = []
        preds = []
        losses = []
        for _, data in DataLoader(dataset, 2048):
            labels += data[0]
            batch_size = len(data[0])
            feed_dict = {
                self.uin: data[1],
                self.ulen: data[2],
                self.iin: data[3],
                self.ilen: data[4],
                self.label: data[0],
                self.batch_size: batch_size,
                self.keep_prob: 1.,
            }
            loss, pred = self.sess.run(fetches=[self.cross_entropy, self.prediction], feed_dict=feed_dict)
            losses.append(loss)
            preds += pred.tolist()
        test_auc = roc_auc_score(labels, preds)
        test_loss = log_loss(labels, preds)
        # f = open('./log', 'a')
        # f.write(str(preds[:100])+'\n')
        print("%s\tSTEP: %s\tLOSS: %.4f\tAUC: %.4f" % (prefix, step, test_loss, test_auc))
        # print(preds[:100])
        # print(losses[:100])
        result = [test_loss, test_auc]
        self.log(step, result, prefix)
        return test_auc

    def train_short(self, epochs=100, p=100):
        step = 0
        flag = True
        losses = []
        for epoch in range(epochs):
            if not flag:
                break
            print('---------------epoch %s---------------' % epoch)
            # random.shuffle(self.trainset)
            start = time.time()
            for _, data in CroppedLoader(self.trainset, self.batchsize):
                step += 1
                batch_size = len(data[0])
                feed_dict = {
                    self.uin: data[2],
                    self.ulen: data[4],
                    self.iin: data[6],
                    self.ilen: data[8],
                    self.label: data[0],
                    self.batch_size: batch_size,
                    self.keep_prob: 0.5,
                }
                self.sess.run(fetches=[self.train_step], feed_dict=feed_dict)
                if step % p == 0:
                    self.eval_short(step, self.trainset, 'train')
                    losses.append(self.eval_short(step, self.testset, 'test'))
                    if len(losses) >= 3 and losses[-1] < losses[-2] < losses[-3]:
                        flag = False
                        break
                    if len(losses) >= 3 and losses[-1] > losses[-2] and losses[-1] > losses[-3]:
                        self.save_model()
            train_time = time.time() - start
            print("Training time: %.4f" % train_time)
        if flag:
            self.save_model()

    def eval_short(self, step, dataset, prefix):
        labels = []
        preds = []
        losses = []
        for _, data in CroppedLoader(dataset, 2048):
            labels += data[0]
            batch_size = len(data[0])
            feed_dict = {
                self.uin: data[2],
                self.ulen: data[4],
                self.iin: data[6],
                self.ilen: data[8],
                self.label: data[0],
                self.batch_size: batch_size,
                self.keep_prob: 1.,
            }
            loss, pred = self.sess.run(fetches=[self.cross_entropy, self.prediction], feed_dict=feed_dict)
            losses.append(loss)
            preds += pred.tolist()
        test_auc = roc_auc_score(labels, preds)
        test_loss = log_loss(labels, preds)
        # f = open('./log', 'a')
        # f.write(str(preds[:100])+'\n')
        print("%s\tSTEP: %s\tLOSS: %.4f\tAUC: %.4f" % (prefix, step, test_loss, test_auc))
        # print(preds[:100])
        # print(losses[:100])
        result = [test_loss, test_auc]
        self.log(step, result, prefix)
        return test_auc


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("PLEASE INPUT [DATASET] and [GPU]")
        sys.exit(0)
    dataset = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

    if dataset == 'taobao':
        fin = open('../data/taobao/dataset.pkl', 'rb')
        train_set = pkl.load(fin)
        test_set = pkl.load(fin)
        feature_size = pkl.load(fin)+1
        fin.close()
        model = RUM('model/rum_taobao/', train_set, test_set, feature_size=feature_size,
                    learning_rate=1e-3, batchsize=10, hidden_size=16, embedding_size=18, beta=0, feature_number=5,
                    user=True, item=False, user_feature_number=4, item_feature_number=3, item_max_len=35, user_max_len=300)
        model.train(5)
    elif dataset == 'amazon':
        fin = open('../data/amazon/dataset.pkl', 'rb')
        train_set = pkl.load(fin)
        test_set = pkl.load(fin)
        feature_size = pkl.load(fin)+1
        fin.close()
        model = RUM('model/rum_amazon/', train_set, test_set, feature_size=feature_size,
                    learning_rate=1e-3, batchsize=10, hidden_size=16, embedding_size=8, beta=0, feature_number=3,
                    user=True, item=False, user_feature_number=3, item_feature_number=2, item_max_len=100, user_max_len=100)
        model.train(5)








    
