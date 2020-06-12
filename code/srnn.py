import cPickle as pkl
import os
import random

import tensorflow as tf
from keras.layers import GRU
from sklearn.metrics import *
import math

from data_loader import *
from ntm_cell import NTMCell

# CROP_PKL_PATH = '../../data/amazon/Books/dataset_crop.pkl'
CROP_PKL_PATH = '../../data/taobao/dataset_crop.pkl'
# PADDING_PKL_PATH = '../../data/amazon/Electronics/dataset_fp.pkl'
random.seed(42)


class BasicModel:
    def __init__(self, path, trainset, testset, input_dim=1, output_dim=1, learning_rate=0.001, batchsize=256):
        self.graph = tf.Graph()
        self._path = path
        self.trainset = trainset
        self.testset = testset
        self._save_path, self._logs_path = None, None
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        with self.graph.as_default():
            self.define_inputs()
            self.build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = self._path
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def define_inputs(self):
        raise NotImplementedError

    def build_graph(self):
        raise NotImplementedError

    def _initialize_session(self, set_logs=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)
        if set_logs:
            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(self._path + 'logs', graph=self.graph)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open("%s/%s" % (
            self.logs_path, str(self.learning_rate)),
                    'a')
        fout.write(s + '\n')

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)


class srnn_dmem(BasicModel):
    def __init__(self, path, trainset, testset, input_dim, user_mem_maxlen, item_mem_maxlen, user_que_maxlen,
                 item_que_maxlen,
                 feature_size, user, item, user_feature_number=3, item_feature_number=2, learning_rate=1e-5,
                 hidden_size=32,
                 embedding_size=10, batchsize=512, beta=1e-4, hops=1, startpoint=100):
        self.user_mem_maxlen = user_mem_maxlen
        self.item_mem_maxlen = item_mem_maxlen
        self.user_que_maxlen = user_que_maxlen
        self.item_que_maxlen = item_que_maxlen

        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.user_feature_number = user_feature_number
        self.item_feature_number = item_feature_number
        self.beta = beta
        self.hops = hops
        self.user = user
        self.item = item
        self.startpoint = startpoint
        BasicModel.__init__(self, path, trainset, testset, input_dim, 1, learning_rate, batchsize)

    def define_inputs(self):
        self.user_mem = tf.placeholder(tf.int32, shape=[None, self.user_mem_maxlen, self.user_feature_number])
        self.user_que = tf.placeholder(tf.int32, shape=[None, self.user_que_maxlen, self.user_feature_number])
        self.item_mem = tf.placeholder(tf.int32, shape=[None, self.item_mem_maxlen, self.item_feature_number])
        self.item_que = tf.placeholder(tf.int32, shape=[None, self.item_que_maxlen, self.item_feature_number])
        self.user_mem_len = tf.placeholder(tf.int32, shape=[None, ])
        self.user_que_len = tf.placeholder(tf.int32, shape=[None, ])
        self.item_mem_len = tf.placeholder(tf.int32, shape=[None, ])
        self.item_que_len = tf.placeholder(tf.int32, shape=[None, ])
        self.label = tf.placeholder(tf.int32, shape=[None, ])
        self.keep_prob = tf.placeholder(tf.float32)

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
        try:
            self.cross_entropy += 1e-6 * self.memory_loss
        except:
            pass
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.cross_entropy)

    def attention(self, key, value, query):
        batchsize, length, dim = tf.shape(key)[0], tf.shape(key)[1], tf.shape(key)[2]
        query = tf.reshape(query, [batchsize, 1, dim])
        weight = tf.nn.softmax(tf.reduce_sum(tf.multiply(key, query), axis=2))
        weight0 = weight
        weight = tf.expand_dims(weight, -1)
        output = tf.reduce_sum(tf.multiply(value, weight), axis=1)
        return output, weight0

    def build_memory(self, input, embedding_size):
        input1 = tf.reshape(input, [-1, 3, embedding_size])
        gru1 = GRU(self.hidden_size, return_sequences=False, use_bias=False)

        embed1 = gru1(input1)
        input2 = tf.reshape(embed1, [-1, 5, self.hidden_size])
        embed1 = tf.reshape(embed1, [-1, 30, self.hidden_size])
        gru2 = GRU(self.hidden_size, return_sequences=False, use_bias=False)

        embed2 = gru2(input2)
        input3 = tf.reshape(embed2, [-1, 6, self.hidden_size])
        embed2 = tf.reshape(embed2, [-1, 6, self.hidden_size])
        gru3 = GRU(self.hidden_size, return_sequences=False, use_bias=False)
        embed3 = gru3(input3)

        memory1 = tf.gather(embed1, [29], axis=1)
        memory2 = tf.gather(embed2, [5], axis=1)
        memory3 = tf.expand_dims(embed3, axis=1)
        memory = tf.concat([memory1, memory2, memory3], axis=1)

        return memory

    def build_repre(self, mem_input, que_input, seqlen, embedding_size, hops=1):
        # mem_input = tf.reshape(mem_input, shape=(-1, 5, 8, 2, embedding_size))
        memory = self.build_memory(mem_input, embedding_size)
        with tf.variable_scope('short'):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            _, repre = tf.nn.dynamic_rnn(cell, que_input, sequence_length=seqlen, dtype=tf.float32)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        # W = tf.get_variable('emb', [self.hidden_size, self.hidden_size], initializer=self.weights_initializer)
        # query = tf.matmul(repre, W)
        query = repre
        weights = []

        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query)
            weights.append(weight)
            query = tf.matmul(query, H) + read

        # def cond(query, H, memory, hop, hops):
        #     return hop < hops
        #
        # def one_hop(query, H, memory, hop, hops):
        #     read = self.attention(memory, memory, query)
        #     query = tf.matmul(query, H) + read
        #     hop += 1
        #     return query, H, memory, hop, hops
        #
        # query, H, memory, hop, hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, hops])

        return tf.concat([query, repre], axis=1), repre, weights

    def embedding(self):
        emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])
        uid_mask_array = [[0.]] + [[1.]] * (self.feature_size - 1)
        mask_lookup_table = tf.get_variable("mask_lookup_table",
                                            initializer=uid_mask_array,
                                            trainable=False)
        mem_uin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.user_mem) * tf.nn.embedding_lookup(
            mask_lookup_table, self.user_mem),
                             [-1, self.user_mem_maxlen, self.user_feature_number * self.embedding_size])
        que_uin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.user_que) * tf.nn.embedding_lookup(
            mask_lookup_table, self.user_que),
                             [-1, self.user_que_maxlen, self.user_feature_number * self.embedding_size])
        mem_iin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.item_mem) * tf.nn.embedding_lookup(
            mask_lookup_table, self.item_mem),
                             [-1, self.item_mem_maxlen, self.item_feature_number * self.embedding_size])
        que_iin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.item_que) * tf.nn.embedding_lookup(
            mask_lookup_table, self.item_que),
                             [-1, self.item_que_maxlen, self.item_feature_number * self.embedding_size])

        return mem_uin, que_uin, mem_iin, que_iin

    def build_graph(self):
        with tf.variable_scope('short'):
            mem_uin, que_uin, mem_iin, que_iin = self.embedding()

        with tf.variable_scope('norm'):
            with tf.variable_scope('user_repre'):
                user_repre, user_repre_short, self.user_weights = self.build_repre(mem_uin, que_uin, self.user_que_len,
                                                                                   self.user_feature_number * self.embedding_size,
                                                                                   hops=self.hops)

            with tf.variable_scope('item_repre'):
                item_repre, item_repre_short, self.item_weights = self.build_repre(mem_iin, que_iin, self.item_que_len,
                                                                                   self.item_feature_number * self.embedding_size,
                                                                                   hops=self.hops)

            if self.user and self.item:
                repre = tf.concat([user_repre, item_repre], axis=1)
            elif self.user:
                repre = user_repre
            else:
                repre = item_repre

            with tf.variable_scope('short'):
                bn1 = tf.layers.batch_normalization(inputs=repre, name='bn1')
                fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.elu, name='fc1')
                dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
                fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.elu, name='fc2')
                dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
                fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
                self.prediction = tf.reshape(tf.nn.sigmoid(fc3), [-1, ])

        global_step = tf.Variable(0, trainable=False)
        log_loss = tf.losses.log_loss(self.label, self.prediction)
        log_loss_short = tf.losses.log_loss(self.label, self.prediction)
        self.cross_entropy = log_loss
        self.cross_entropy_short = log_loss_short
        for v in tf.trainable_variables():
            if 'emb' not in v.name and 'bias' not in v.name:
                self.cross_entropy += self.beta * tf.nn.l2_loss(v)
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*short'):
            if 'emb' not in v.name and 'bias' not in v.name:
                self.cross_entropy_short += self.beta * tf.nn.l2_loss(v)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_step_short = optimizer.minimize(self.cross_entropy_short,
                                                   var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                              scope='.*short'), global_step=global_step)
        self.train_step = optimizer.minimize(self.cross_entropy, global_step=global_step)

    def train(self, epochs=100):
        step = 0
        flag = True
        losses = []
        for epoch in range(epochs):
            if not flag:
                break
            print('---------------epoch %s---------------' % epoch)
            random.shuffle(self.trainset)
            start = time.time()
            for _, data in CroppedLoader(self.trainset, self.batchsize):
                step += 1
                feed_dict = {
                    self.user_mem: data[1],
                    self.user_que: data[2],
                    self.item_mem: data[5],
                    self.item_que: data[6],
                    self.user_mem_len: data[3],
                    self.user_que_len: data[4],
                    self.item_mem_len: data[7],
                    self.item_que_len: data[8],
                    self.label: data[0],
                    self.keep_prob: 0.5,
                }
                self.sess.run(fetches=[self.train_step], feed_dict=feed_dict)
                if step % 100 == 0:
                    self.eval(step, self.trainset, 'train', epoch)
                    losses.append(self.eval(step, self.testset, 'test', epoch))
                    if len(losses) >= 3 and losses[-1] < losses[-2] < losses[-3]:
                        flag = False
                        break
                    if len(losses) >= 3 and losses[-1] > losses[-2] and losses[-1] > losses[-3]:
                        self.save_model()
            traintime = time.time() - start
            print("Training time: %.4f" % (traintime))
        if flag:
            self.save_model()

    def eval(self, step, dataset, prefix, epoch):
        labels = []
        preds = []
        losses = []
        for _, data in CroppedLoader(dataset, 2048):
            labels += data[0]
            feed_dict = {
                self.user_mem: data[1],
                self.user_que: data[2],
                self.item_mem: data[5],
                self.item_que: data[6],
                self.user_mem_len: data[3],
                self.user_que_len: data[4],
                self.item_mem_len: data[7],
                self.item_que_len: data[8],
                self.label: data[0],
                self.keep_prob: 1.,
            }
            loss, pred = self.sess.run(fetches=[self.cross_entropy, self.prediction], feed_dict=feed_dict)
            losses.append(loss)
            preds += pred.tolist()
        testloss = np.average(losses)
        testauc = roc_auc_score(labels, preds)
        testloss = log_loss(labels, preds)
        # f = open('./log', 'a')
        # f.write(str(preds[:100])+'\n')
        print("%s\tSTEP: %s\tLOSS: %.4f\tAUC: %.4f" % (prefix, step, testloss, testauc))
        # print(preds[:100])
        # print(losses[:100])
        result = [testloss, testauc]
        self.log(step, result, prefix)
        return testauc

    def get_weight(self):
        user_weights = []
        item_weights = []
        for _, data in CroppedLoader(self.trainset, self.batchsize):
            feed_dict = {
                self.user_mem: data[1],
                self.user_que: data[2],
                self.item_mem: data[5],
                self.item_que: data[6],
                self.user_mem_len: data[3],
                self.user_que_len: data[4],
                self.item_mem_len: data[7],
                self.item_que_len: data[8],
                self.label: data[0],
                self.keep_prob: 1.,
            }
            user_weight, item_weight = self.sess.run(fetches=[self.user_weights, self.item_weights],
                                                     feed_dict=feed_dict)
            user_weights.append(user_weight)
            item_weights.append(item_weight)
        for _, data in CroppedLoader(self.testset, self.batchsize):
            feed_dict = {
                self.user_mem: data[1],
                self.user_que: data[2],
                self.item_mem: data[5],
                self.item_que: data[6],
                self.user_mem_len: data[3],
                self.user_que_len: data[4],
                self.item_mem_len: data[7],
                self.item_que_len: data[8],
                self.label: data[0],
                self.keep_prob: 1.,
            }
            user_weight, item_weight = self.sess.run(fetches=[self.user_weights, self.item_weights],
                                                     feed_dict=feed_dict)
            user_weights.append(user_weight)
            item_weights.append(item_weight)
        user_weights = np.array(user_weights)
        item_weights = np.array(item_weights)
        print(user_weights.shape)
        print(item_weights.shape)
        np.save(self._path + '/user_weights', user_weights)
        np.save(self._path + '/item_weights', item_weights)


class shan(srnn_dmem):
    def __init__(self, path, trainset, testset, input_dim, user_mem_maxlen, item_mem_maxlen, user_que_maxlen,
                 item_que_maxlen,
                 feature_size, emb_init, user=False, item=False, dual=True, user_feature_number=3,
                 item_feature_number=2,
                 learning_rate=1e-5,
                 hidden_size=32, embedding_size=10, batchsize=512, beta=1e-4, hop=1, startpoint=100):
        self.dual = dual
        self.emb_init = emb_init
        srnn_dmem.__init__(self, path, trainset, testset, input_dim, user_mem_maxlen, item_mem_maxlen, user_que_maxlen,
                           item_que_maxlen,
                           feature_size, user, item, user_feature_number, item_feature_number, learning_rate,
                           hidden_size,
                           embedding_size, batchsize, beta, hop, startpoint)

    def build_fc_net(self, inp, keep_prob=0.8):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.prediction = tf.reshape(tf.nn.sigmoid(fc3), [-1, ])

        # loss
        self.log_loss = tf.losses.log_loss(self.label, self.prediction)
        self.cross_entropy = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.cross_entropy += self.beta * tf.nn.l2_loss(v)

        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.cross_entropy)

    def attention_v2(self, key, value, query, mask):
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)  # [B, Dk]
        fc1 = tf.layers.dense(key, k_dim, activation=tf.nn.relu)  # [B, T, Dk]
        queries = tf.expand_dims(query, 1)  # [B, 1, Dk]
        product = tf.reduce_sum((queries * key), axis=-1)  # [B, T]

        mask = tf.reshape(tf.equal(mask, tf.ones_like(mask)), [-1, max_len])  # [B, T]
        paddings = tf.ones_like(product) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.where(mask, product, paddings))  # [B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2))  # [B, T, Dv==Dk]
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)  # [B, Dv==Dk]

        return atten_output_sum, atten_output, score

    def prelu(self, _x, scope=None):
        """parametric ReLU activation"""
        with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
            _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                     dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

    def attention(self, key, value, query, mask):
        # key, value: [B, T, Dk], query: [B, Dq], mask: [B, T, 1]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        query = self.prelu(query)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        inp = tf.concat([queries, key, queries - key, queries * key], axis=-1)
        fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None)  # [B, T, 1]

        mask = tf.equal(mask, tf.ones_like(mask))  # [B, T, 1]
        paddings = tf.ones_like(fc3) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.reshape(tf.where(mask, fc3, paddings), [-1, max_len]))  # [B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2))
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return atten_output_sum, atten_output, score

    def build_graph(self):
        self.weights_initializer = tf.random_normal_initializer(stddev=0.1)
        self.biases_initializer = tf.zeros_initializer
        with tf.variable_scope('short'):
            if self.emb_init is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=self.emb_init)

            uid_mask_array = [[0.]] + [[1.]] * (self.feature_size - 1)
            mask_lookup_table = tf.get_variable("mask_lookup_table",
                                                initializer=uid_mask_array,
                                                trainable=False)
            mem_uin = tf.reshape(tf.nn.embedding_lookup(self.emb_mtx, self.user_mem) * tf.nn.embedding_lookup(
                mask_lookup_table, self.user_mem),
                                 [-1, self.user_mem_maxlen, self.user_feature_number * self.embedding_size])
            que_uin = tf.reshape(tf.nn.embedding_lookup(self.emb_mtx, self.user_que) * tf.nn.embedding_lookup(
                mask_lookup_table, self.user_que),
                                 [-1, self.user_que_maxlen, self.user_feature_number * self.embedding_size])
            mem_iin = tf.reshape(tf.nn.embedding_lookup(self.emb_mtx, self.item_mem) * tf.nn.embedding_lookup(
                mask_lookup_table, self.item_mem),
                                 [-1, self.item_mem_maxlen, self.item_feature_number * self.embedding_size])
            que_iin = tf.reshape(tf.nn.embedding_lookup(self.emb_mtx, self.item_que) * tf.nn.embedding_lookup(
                mask_lookup_table, self.item_que),
                                 [-1, self.item_que_maxlen, self.item_feature_number * self.embedding_size])

        # +1 is for long term attention weighted sum
        user_mask_mem = tf.expand_dims(tf.sequence_mask(self.user_mem_len, tf.shape(mem_uin)[1], dtype=tf.float32), 2)
        user_mask_mem_1 = tf.expand_dims(
            tf.sequence_mask(self.user_mem_len - 1, tf.shape(mem_uin)[1], dtype=tf.float32), 2)
        user_mask_que = tf.expand_dims(tf.sequence_mask(self.user_que_len, tf.shape(que_uin)[1], dtype=tf.float32), 2)
        user_mask_que_1 = tf.expand_dims(
            tf.sequence_mask(self.user_que_len - 1, tf.shape(que_uin)[1], dtype=tf.float32), 2)
        user_short_mask = tf.expand_dims(
            tf.sequence_mask(self.user_que_len + 1, tf.shape(que_uin)[1] + 1, dtype=tf.float32), 2)

        item_mask_mem = tf.expand_dims(tf.sequence_mask(self.item_mem_len, tf.shape(mem_iin)[1], dtype=tf.float32), 2)
        item_mask_mem_1 = tf.expand_dims(
            tf.sequence_mask(self.item_mem_len - 1, tf.shape(mem_iin)[1], dtype=tf.float32), 2)
        item_mask_que = tf.expand_dims(tf.sequence_mask(self.item_que_len, tf.shape(que_iin)[1], dtype=tf.float32), 2)
        item_mask_que_1 = tf.expand_dims(
            tf.sequence_mask(self.item_que_len - 1, tf.shape(que_iin)[1], dtype=tf.float32), 2)
        item_short_mask = tf.expand_dims(
            tf.sequence_mask(self.item_que_len + 1, tf.shape(que_iin)[1] + 1, dtype=tf.float32), 2)

        target_item = tf.reduce_sum((user_mask_que - user_mask_que_1) * que_uin, axis=1)  # [B, F*EMB]
        target_user = tf.reduce_sum((item_mask_que - item_mask_que_1) * que_iin, axis=1)

        # with tf.variable_scope('user_long_atten'):
        #     atten_sum_user, _, _ = self.attention(mem_uin * user_mask_mem_1, mem_uin * user_mask_mem_1, target_user,
        #                                           user_mask_mem_1)
        #     short_rep_user = tf.concat([tf.expand_dims(atten_sum_user, 1), que_uin], axis=1)
        with tf.name_scope('item_short_atten'):
            user_part, _, _ = self.attention(que_uin, que_uin, target_user, user_mask_que)

        # with tf.variable_scope('item_long_atten'):
        #     atten_sum_item, _, _ = self.attention(mem_iin * item_mask_mem_1, mem_iin * item_mask_mem_1, target_item,
        #                                           item_mask_mem_1)
        #     short_rep_item = tf.concat([tf.expand_dims(atten_sum_item, 1), que_iin], axis=1)
        with tf.name_scope('item_short_atten'):
            item_part, _, _ = self.attention(que_iin, que_iin, target_item, item_mask_que)

        if self.dual:
            inp = tf.concat([user_part, item_part, target_user, target_item], axis=1)
        else:
            inp = tf.concat([user_part, target_item], axis=1)

        # fully connected layer
        self.build_fc_net(inp)

    def train(self, epochs=100):
        step = 0
        flag = True
        losses = []
        for epoch in range(epochs):
            if not flag:
                break
            print('---------------epoch %s---------------' % epoch)
            random.shuffle(self.trainset)
            start = time.time()
            for _, data in CroppedLoader(self.trainset, self.batchsize):
                step += 1
                feed_dict = {
                    self.user_mem: data[1],
                    self.user_que: data[2],
                    self.item_mem: data[5],
                    self.item_que: data[6],
                    self.user_mem_len: data[3],
                    self.user_que_len: data[4],
                    self.item_mem_len: data[7],
                    self.item_que_len: data[8],
                    self.label: data[0],
                    self.keep_prob: 0.5,
                }
                if step <= self.startpoint:
                    self.sess.run(fetches=[self.train_step_short], feed_dict=feed_dict)
                else:
                    self.sess.run(fetches=[self.train_step], feed_dict=feed_dict)
                if step % 100 == 0:
                    self.eval(step, self.trainset, 'train', epoch)
                    losses.append(self.eval(step, self.testset, 'test', epoch))
                    if len(losses) >= 3 and losses[-1] < losses[-2] < losses[-3]:
                        flag = False
                        break
            traintime = time.time() - start
            print("Training time: %.4f" % (traintime))

    def eval(self, step, dataset, prefix, epoch):
        labels = []
        preds = []
        losses = []
        for _, data in CroppedLoader(dataset, 2048):
            labels += data[0]
            feed_dict = {
                self.user_mem: data[1],
                self.user_que: data[2],
                self.item_mem: data[5],
                self.item_que: data[6],
                self.user_mem_len: data[3],
                self.user_que_len: data[4],
                self.item_mem_len: data[7],
                self.item_que_len: data[8],
                self.label: data[0],
                self.keep_prob: 1.,
            }
            if step > self.startpoint:
                loss, pred = self.sess.run(fetches=[self.cross_entropy, self.prediction],
                                           feed_dict=feed_dict)
            else:
                loss, pred = self.sess.run(fetches=[self.cross_entropy_short, self.prediction],
                                           feed_dict=feed_dict)
            losses.append(loss)
            preds += pred.tolist()
        testloss = np.average(losses)
        testauc = roc_auc_score(labels, preds)
        testloss = log_loss(labels, preds)
        # f = open('./log', 'a')
        # f.write(str(preds[:100])+'\n')
        print("%s\tSTEP: %s\tLOSS: %.4f\tAUC: %.4f" % (prefix, step, testloss, testauc))
        # print(preds[:100])
        # print(losses[:100])
        result = [testloss, testauc]
        self.log(step, result, prefix)
        return testauc


class Memory_Ind(BasicModel):
    def __init__(self, path, trainset, testset, input_dim, maxlen, memory_maxlen, feature_size, user_feature_size,
                 emb_initializer=None,
                 learning_rate=1e-5, hidden_size=32, embedding_size=16, batchsize=2500, beta=1e-4, hop=1):
        self.hops = hop
        self.feature_size = feature_size
        self.maxlen = maxlen
        self.memory_maxlen = memory_maxlen
        self.hidden_size = hidden_size
        self.emb_initializer = emb_initializer
        self.embedding_size = embedding_size
        self.beta = beta
        self.user_feature_size = user_feature_size
        BasicModel.__init__(self, path, trainset, testset, input_dim, 1, learning_rate, batchsize)

    # def attention(self, key, value, query):
    #     batchsize, length, dim = tf.shape(key)[0], tf.shape(key)[1], tf.shape(key)[2]
    #     query = tf.reshape(query, [batchsize, 1, dim])
    #     weight = tf.nn.softmax(tf.reduce_sum(tf.multiply(key, query), axis=2))
    #     weight0 = weight
    #     weight = tf.expand_dims(weight, -1)
    #     output = tf.reduce_sum(tf.multiply(value, weight), axis=1)
    #     return output, weight0

    def attention(self, key, value, query):
        # key, value: [B, T, Dk], query: [B, Dq], mask: [B, T, 1]
        # _, max_len, k_dim = key.get_shape().as_list()
        # query = tf.layers.dense(query, k_dim, activation=None)
        # query = self.prelu(query)
        k = key.get_shape().as_list()[1]
        queries = tf.tile(tf.expand_dims(query, 1), [1, k, 1])  # [B, T, Dk]
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

    def define_inputs(self):
        self.user_id = tf.placeholder(tf.int32, shape=[None, ])
        self.sequence = tf.placeholder(tf.int32, shape=[None, self.maxlen, self.input_dim])
        self.label = tf.placeholder(tf.int32, shape=[None, ])
        self.index = tf.placeholder(tf.int32, shape=())
        self.keep_prob = tf.placeholder(tf.float32)
        self.weight_list = tf.placeholder(tf.float32, [None, 7])

    def build_fc_net(self, inp, reg=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.elu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.elu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.prediction = tf.reshape(tf.nn.sigmoid(fc3), [-1, ])

        # loss
        self.log_loss = tf.losses.log_loss(self.label, self.prediction)
        self.cross_entropy = self.log_loss
        for v in tf.trainable_variables():
            self.cross_entropy += self.beta * tf.nn.l2_loss(v)
        if reg:
            self.cross_entropy += 1e-5 * self.memory_loss
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.cross_entropy)

    def init_memory(self, input, states, cells):
        i = tf.constant(0)
        memory = []

        def next_layer(outputs):
            outputs = tf.reshape(outputs, [-1, self.maxlen / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2),
                               [-1, self.maxlen / 2 ** (i + 1), self.hidden_size])
            return input

        def cond(i, input, memory):
            return i < 9

        def body(i, input, memory):
            cell = cells[0]
            del cells[0]
            outputs, state = tf.nn.dynamic_rnn(cell, input, initial_state=tf.gather(states, i, axis=1))
            input = tf.case([(i < 8, lambda: next_layer(outputs))])
            memory.append(state)
            i += 1
            return i, input, memory

        _, _, memory = tf.while_loop(cond, body, [i, input, memory], shape_invariants=[i.get_shape(), tf.TensorShape(
            [self.batchsize, self.hidden_size]), tf.TensorShape([self.batchsize, None, self.hidden_size])])
        return tf.concat(memory, axis=1)

    def incremental_update(self, input, states, cells):
        i = tf.constant(0)
        memory = tf.zeros(shape=[self.batchsize, 1, self.hidden_size], dtype=tf.float32)
        input = tf.layers.flatten(tf.gather(input, [256], axis=1))

        def cond2(i, memory, input):
            return self.index % (2 ** i) == 0

        def body2(i, memory, input):
            cell = cells[0]
            del cells[0]
            output, state = cell.__call__(input, tf.gather(states, i, axis=1))
            input = state
            state = tf.expand_dims(state, axis=1)
            memory = tf.concat([memory, state], axis=1)

            i += 1
            return i, memory, input

        _, memory, _ = tf.while_loop(cond2, body2, [i, memory, input],
                                     shape_invariants=[i.get_shape(),
                                                       tf.TensorShape([None, None, self.hidden_size]),
                                                       tf.TensorShape([None, self.hidden_size])])
        return memory[:, 1:, :]

    def build_memory(self, input, states, index):
        cells = []
        for _ in range(9):
            cells.append(tf.nn.rnn_cell.GRUCell(self.hidden_size))
        index = index - 256
        return tf.case([(tf.equal(index, 0), lambda: self.init_memory(input, states, cells))],
                       default=lambda: self.incremental_update(input, states, cells))

    def build_repre(self, input, states):
        query = tf.layers.flatten(input[:, 256, :])
        mem_input = input[:, :-1, :]
        memory = self.build_memory(mem_input, states, self.index)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])

        def cond(query, H, memory, hop, hops):
            return hop < self.hops

        def one_hop(query, H, memory, hop, hops):
            read = self.attention(memory, memory, query)
            query = tf.matmul(query, H) + read
            hop += 1
            return query, H, memory, hop, hops

        query, H, memory, hop, self.hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, self.hops])

        return query, memory

    def build_graph(self):
        with tf.name_scope('Embedding'):
            if self.emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=self.emb_initializer)
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])

            input = tf.reshape(
                tf.nn.embedding_lookup(self.emb_mtx, self.sequence),
                [-1, self.maxlen + 1, self.input_dim * self.embedding_size])

        with tf.variable_scope('repre'):
            state_store = tf.get_variable('state_mtx', [self.user_feature_size, 9 * self.hidden_size])
            states = tf.split(tf.nn.embedding_lookup(state_store, self.user_id), 9, axis=1)
            repre, memory = self.build_repre(input, states)

        with tf.variable_scope('update'):
            self.update_op = tf.scatter_nd_update(state_store, tf.expand_dims(self.user_id, axis=-1),
                                                  tf.layers.flatten(memory))

        self.build_fc_net(repre)

    def train(self, epochs=1):
        step = 0
        flag = True
        losses = []
        for epoch in range(epochs):
            if not flag:
                break
            print('---------------epoch %s---------------' % epoch)
            start = time.time()
            for data in DataLoader_unsort(self.trainset, self.batchsize):
                step += 1
                feed_dict = {
                    self.index: data[0],
                    self.user_id: data[1],
                    self.label: data[2],
                    self.sequence: data[3],
                    self.keep_prob: 0.5,
                }
                _, loss = self.sess.run(fetches=[self.train_step, self.cross_entropy],
                                        feed_dict=feed_dict)
                if step % 5 == 0:
                    losses.append(self.eval(step, self.testset, 'test'))
                    if len(losses) >= 3 and losses[-1] < losses[-2] < losses[-3]:
                        flag = False
                        break
                if step % 20 == 0:
                    self.eval(step, self.trainset, 'train')
                # if len(losses) >= 3 and losses[-1] > losses[-2] and losses[-1] > losses[-3]:
                # self.save_model()
        #
        # if flag:
        #     self.save_model()

    def eval(self, step, dataset, prefix):
        labels = []
        preds = []
        losses = []
        for data in DataLoader_unsort(dataset, self.batchsize):
            labels += data[2]
            feed_dict = {
                self.index: data[0],
                self.user_id: data[1],
                self.label: data[2],
                self.sequence: data[3],
                self.keep_prob: 1.,
            }
            loss, pred = self.sess.run(fetches=[self.log_loss, self.prediction],
                                       feed_dict=feed_dict)
            losses.append(loss)
            preds += pred.tolist()
        testloss = np.average(losses)
        testauc = roc_auc_score(labels, preds)
        print("%s\tSTEP: %s\tLOSS: %.4f\tAUC: %.4f" % (prefix, step, testloss, testauc))
        result = [testloss, testauc]
        self.log(step, result, prefix)
        return testauc


class Memory_Noupdate(Memory_Ind):
    def __init__(self, path, trainset, testset, input_dim, maxlen, memory_maxlen, feature_size, user_feature_size,
                 emb_initializer=None,
                 learning_rate=1e-5, hidden_size=32, embedding_size=16, batchsize=2500, beta=1e-4, hop=1, layers=6):
        self.layers = layers
        Memory_Ind.__init__(self, path, trainset, testset, input_dim, maxlen, memory_maxlen, feature_size,
                            user_feature_size,
                            emb_initializer=emb_initializer,
                            learning_rate=learning_rate, hidden_size=hidden_size, embedding_size=embedding_size,
                            batchsize=batchsize, beta=beta, hop=hop)

    def build_repre(self, input):
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        _, query = tf.nn.dynamic_rnn(cell, input[:, self.memory_maxlen:, :], dtype=tf.float32)
        last = query
        print(last)
        mem_input = input[:, :self.memory_maxlen, :]
        memory = self.build_memory(mem_input)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])

        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query)
            query = tf.matmul(query, H) + read

        return tf.concat([query, last], axis=-1), weight

    def build_memory(self, input):
        memory = []
        with tf.variable_scope('GRU0'):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        memory.append(tf.expand_dims(state, axis=1))
        outputs = tf.reshape(outputs, [-1, self.memory_maxlen / 3, 3, self.hidden_size])
        input = tf.reshape(tf.gather(outputs, [2], axis=2),
                           [-1, self.memory_maxlen / 3, self.hidden_size])

        for i in range(self.layers):
            with tf.variable_scope('GRU%s' % (i + 1)):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            outputs = tf.reshape(outputs, [-1, self.memory_maxlen / 3 / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2),
                               [-1, self.memory_maxlen / 3 / 2 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_graph(self):
        with tf.variable_scope('Embedding'):
            if self.emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=self.emb_initializer)
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])

            input = tf.reshape(
                tf.nn.embedding_lookup(self.emb_mtx, self.sequence),
                [-1, self.maxlen, self.input_dim * self.embedding_size])

        with tf.variable_scope('repre'):
            repre, self.weight = self.build_repre(input)

        self.build_fc_net(repre)

    def get_weights(self):
        weights = []
        ids = []
        preds = []
        for data in DataLoader_unsort(self.trainset, self.batchsize):
            feed_dict = {
                self.index: data[0],
                self.user_id: data[1],
                self.label: data[2],
                self.sequence: data[3],
                self.keep_prob: 1.,
                self.weight_list: np.zeros([self.batchsize, 6], np.float32)
            }
            seq = data[3]
            id = seq[:, -1, 1]
            weight, pred = self.sess.run(fetches=[self.weight, self.prediction],
                                         feed_dict=feed_dict)
            weights += weight.tolist()
            ids += id.tolist()
            preds += pred.tolist()
        for data in DataLoader_unsort(self.testset, self.batchsize):
            feed_dict = {
                self.index: data[0],
                self.user_id: data[1],
                self.label: data[2],
                self.sequence: data[3],
                self.keep_prob: 1.,
                self.weight_list: np.zeros([self.batchsize, 6], np.float32)
            }
            seq = data[3]
            id = seq[:, -1, 1]
            weight, pred = self.sess.run(fetches=[self.weight, self.prediction],
                                         feed_dict=feed_dict)
            weights += weight.tolist()
            ids += id.tolist()
            preds += pred.tolist()
        weights = np.array(weights)
        ids = np.array(ids)
        preds = np.array(preds)
        print(weights.shape)
        print(ids.shape)
        print(preds.shape)
        np.save(self._path + '/weights', weights)
        np.save(self._path + '/ids', ids)
        np.save(self._path + '/preds', preds)


class Memory_taobao(srnn_dmem):
    def attention(self, key, value, query, k):
        # key, value: [B, T, Dk], query: [B, Dq], mask: [B, T, 1]
        # _, max_len, k_dim = key.get_shape().as_list()
        # query = tf.layers.dense(query, k_dim, activation=None)
        # query = self.prelu(query)
        k = key.get_shape().as_list()[1]
        queries = tf.tile(tf.expand_dims(query, 1), [1, k, 1])  # [B, T, Dk]
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

    def build_user_memory(self, input):
        memory = []
        for i in range(9):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = NTMCell(1, self.hidden_size, 5, self.hidden_size, 1, 1, output_dim=self.hidden_size)
                # init_state = cell.zero_state(self.batchsize, dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            # memory.append(state[3])
            outputs = tf.reshape(outputs, [-1, self.user_mem_maxlen / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2),
                               [-1, self.user_mem_maxlen / 2 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_item_memory(self, input):
        memory = []
        for i in range(4):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = NTMCell(1, self.hidden_size, 5, self.hidden_size, 1, 1, output_dim=self.hidden_size)
                # init_state = cell.zero_state(self.batchsize, dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            # memory.append(state[3])
            outputs = tf.reshape(outputs, [-1, self.item_mem_maxlen / 3 ** (i + 1), 3, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [2], axis=2),
                               [-1, self.item_mem_maxlen / 3 ** (i + 1), self.hidden_size])
        memory = tf.concat(memory, axis=1)
        return memory

    def build_user_repre(self):
        batchsize = tf.shape(self.que_uin)[0]
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        _, query = tf.nn.dynamic_rnn(cell, self.que_uin, dtype=tf.float32, sequence_length=self.user_que_len)
        index = tf.range(0, batchsize) * self.user_que_maxlen + self.user_que_len - 1
        last = tf.gather(tf.reshape(self.que_uin, [-1, self.user_feature_number * self.embedding_size]), index)
        query = tf.concat([query, last], axis=-1)
        last = query
        memory = self.build_user_memory(self.mem_uin)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        I = tf.get_variable('mapping2',
                            [self.user_feature_number * self.embedding_size + self.hidden_size, self.hidden_size])
        query = tf.matmul(query, I)

        def cond(query, H, memory, hop, hops):
            return hop < self.hops

        def one_hop(query, H, memory, hop, hops):
            read, score = self.attention(memory, memory, query, 9)
            query = tf.matmul(query, H) + read
            hop += 1
            return query, H, memory, hop, hops

        query, H, memory, hop, self.hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, self.hops])

        return tf.concat([query, last], axis=-1)

    def build_item_repre(self):
        batchsize = tf.shape(self.que_iin)[0]
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        _, query = tf.nn.dynamic_rnn(cell, self.que_iin, dtype=tf.float32, sequence_length=self.item_que_len)
        index = tf.range(0, batchsize) * self.item_que_maxlen + self.item_que_len - 1
        last = tf.gather(tf.reshape(self.que_iin, [-1, self.item_feature_number * self.embedding_size]), index)
        query = tf.concat([query, last], axis=-1)
        last = query
        memory = self.build_item_memory(self.mem_iin)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        I = tf.get_variable('mapping2',
                            [self.item_feature_number * self.embedding_size + self.hidden_size, self.hidden_size])
        query = tf.matmul(query, I)

        def cond(query, H, memory, hop, hops):
            return hop < self.hops

        def one_hop(query, H, memory, hop, hops):
            read, score = self.attention(memory, memory, query, 4)
            print(read)
            query = tf.matmul(query, H) + read
            hop += 1
            return query, H, memory, hop, hops

        query, H, memory, hop, self.hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, self.hops])

        return tf.concat([query, last], axis=-1)

    def build_graph(self):
        with tf.variable_scope('embedding'):
            self.mem_uin, self.que_uin, self.mem_iin, self.que_iin = self.embedding()

        with tf.variable_scope('user_repre'):
            user_repre = self.build_user_repre()

        with tf.variable_scope('item_repre'):
            item_repre = self.build_item_repre()

        if self.user:
            if self.item:
                repre = tf.concat([user_repre, item_repre], 1)
            else:
                repre = user_repre
        else:
            repre = item_repre

        self.build_fc_net(repre, self.keep_prob)


class Memory_amazon(Memory_taobao):
    def build_user_memory(self, input):
        memory = []
        lengths = [2, 3, 3, 5, 1]
        maxlen = self.user_mem_maxlen
        for i in range(4):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = NTMCell(1, self.hidden_size, 5, self.hidden_size, read_head_num=1, write_head_num=1, output_dim=self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            # memory.append(state[3])
            maxlen /= lengths[i]
            outputs = tf.reshape(outputs, [-1, maxlen, lengths[i], self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [lengths[i] - 1], axis=2),
                               [-1, maxlen, self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_item_memory(self, input):
        memory = []
        lengths = [2, 3, 3, 5, 1]
        batchsize = input.get_shape().as_list()[0]
        maxlen = self.item_mem_maxlen
        for i in range(4):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = NTMCell(1, self.hidden_size, 5, self.hidden_size, read_head_num=1, write_head_num=1,
                #                output_dim=self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            # memory.append(state[3])
            maxlen /= lengths[i]
            outputs = tf.reshape(outputs, [-1, maxlen, lengths[i], self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [lengths[i] - 1], axis=2),
                               [-1, maxlen, self.hidden_size])
        memory = tf.concat(memory, axis=1)
        return memory


class taobao_noshort(Memory_taobao):
    def __init__(self, path, trainset, testset, input_dim, user_maxlen, item_maxlen,
                 feature_size, user, item, user_feature_number=3, item_feature_number=2, learning_rate=1e-5,
                 hidden_size=32,
                 embedding_size=10, batchsize=512, beta=1e-4, hops=1, startpoint=100):
        self.user_maxlen = user_maxlen
        self.item_maxlen = item_maxlen

        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.user_feature_number = user_feature_number
        self.item_feature_number = item_feature_number
        self.beta = beta
        self.hops = hops
        self.user = user
        self.item = item
        self.startpoint = startpoint
        BasicModel.__init__(self, path, trainset, testset, input_dim, 1, learning_rate, batchsize)

    def define_inputs(self):
        self.user_input = tf.placeholder(tf.int32, shape=[None, self.user_maxlen, self.user_feature_number])
        self.item_input = tf.placeholder(tf.int32, shape=[None, self.item_maxlen, self.item_feature_number])
        self.user_len = tf.placeholder(tf.int32, shape=[None, ])
        self.item_len = tf.placeholder(tf.int32, shape=[None, ])
        self.label = tf.placeholder(tf.int32, shape=[None, ])
        self.keep_prob = tf.placeholder(tf.float32)

    def embedding(self):
        emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])
        uid_mask_array = [[0.]] + [[1.]] * (self.feature_size - 1)
        mask_lookup_table = tf.get_variable("mask_lookup_table",
                                            initializer=uid_mask_array,
                                            trainable=False)
        uin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.user_input) * tf.nn.embedding_lookup(
            mask_lookup_table, self.user_input),
                         [-1, self.user_maxlen, self.user_feature_number * self.embedding_size])

        iin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.item_input) * tf.nn.embedding_lookup(
            mask_lookup_table, self.item_input),
                         [-1, self.item_maxlen, self.item_feature_number * self.embedding_size])

        return uin, iin

    def build_user_memory(self, input):
        memory = []
        lengths = [2, 2, 3, 5, 5, 1]
        len = self.user_maxlen
        for i in range(6):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            len /= lengths[i]
            outputs = tf.reshape(outputs, [-1, len, lengths[i], self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [lengths[i] - 1], axis=2),
                               [-1, len, self.hidden_size])
        memory = tf.concat(memory, axis=1)
        return memory

    def build_item_memory(self, input):
        memory = []
        lengths = [2, 2, 3, 3, 1]
        len = self.item_maxlen
        for i in range(5):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = NTMCell(1, self.hidden_size, 5, self.hidden_size, read_head_num=1, write_head_num=1, output_dim=self.hidden_size)
                init_state = cell.zero_state(self.batchsize, dtype=tf.float32)
                outputs, state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state)
            memory.append(tf.expand_dims(state, axis=1))
            # memory.append(state[3])
            len /= lengths[i]
            outputs = tf.reshape(outputs, [-1, len, lengths[i], self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [lengths[i] - 1], axis=2),
                               [-1, len, self.hidden_size])
        memory = tf.concat(memory, axis=1)
        return memory

    def build_user_repre(self):
        last = self.uin[:, -1, :]
        query = last
        memory = self.build_user_memory(self.uin)
        I = tf.get_variable('input_map', [self.user_feature_number * self.embedding_size, self.hidden_size])
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        query = tf.matmul(query, I)
        weights = []
        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query, 6)
            query = tf.matmul(query, H) + read
            weights.append(weight)

        return tf.concat([query, last], axis=-1), weights

    def build_item_repre(self):
        last = self.iin[:, -1, :]
        query = last
        memory = self.build_user_memory(self.iin)
        I = tf.get_variable('input_map', [self.item_feature_number * self.embedding_size, self.hidden_size])
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        query = tf.matmul(query, I)
        weights = []
        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query, 5)
            query = tf.matmul(query, H) + read
            weights.append(weight)

        return tf.concat([query, last], axis=-1), weights

    def build_graph(self):
        with tf.variable_scope('embedding'):
            self.uin, self.iin = self.embedding()

        with tf.variable_scope('user_repre'):
            user_repre, _ = self.build_user_repre()

        with tf.variable_scope('item_repre'):
            item_repre, _ = self.build_item_repre()

        if self.user:
            if self.item:
                repre = tf.concat([user_repre, item_repre], 1)
            else:
                repre = user_repre
        else:
            repre = item_repre

        print(repre)
        self.build_fc_net(repre, self.keep_prob)

    def train(self, epochs=100):
        step = 0
        flag = True
        losses = []
        for epoch in range(epochs):
            if not flag:
                break
            print('---------------epoch %s---------------' % epoch)
            random.shuffle(self.trainset)
            start = time.time()
            for _, data in DataLoader(self.trainset, self.batchsize):
                step += 1
                feed_dict = {
                    self.user_input: data[1],
                    self.item_input: data[3],
                    self.user_len: data[2],
                    self.item_len: data[4],
                    self.label: data[0],
                    self.keep_prob: 0.5,
                }
                self.sess.run(fetches=[self.train_step], feed_dict=feed_dict)
                if step % 100 == 0:
                    self.eval(step, self.trainset, 'train', epoch)
                    losses.append(self.eval(step, self.testset, 'test', epoch))
                    if len(losses) >= 3 and losses[-1] < losses[-2] < losses[-3]:
                        flag = False
                        break
                    if len(losses) >= 3 and losses[-1] > losses[-2] and losses[-1] > losses[-3]:
                        self.save_model()
            traintime = time.time() - start
            print("Training time: %.4f" % (traintime))
        if flag:
            self.save_model()

    def eval(self, step, dataset, prefix, epoch):
        labels = []
        preds = []
        losses = []
        for _, data in DataLoader(dataset, 2048):
            labels += data[0]
            feed_dict = {
                self.user_input: data[1],
                self.item_input: data[3],
                self.user_len: data[2],
                self.item_len: data[4],
                self.label: data[0],
                self.keep_prob: 1.,
            }
            loss, pred = self.sess.run(fetches=[self.cross_entropy, self.prediction], feed_dict=feed_dict)
            losses.append(loss)
            preds += pred.tolist()
        testloss = np.average(losses)
        testauc = roc_auc_score(labels, preds)
        testloss = log_loss(labels, preds)
        # f = open('./log', 'a')
        # f.write(str(preds[:100])+'\n')
        print("%s\tSTEP: %s\tLOSS: %.4f\tAUC: %.4f" % (prefix, step, testloss, testauc))
        # print(preds[:100])
        # print(losses[:100])
        result = [testloss, testauc]
        self.log(step, result, prefix)
        return testauc


class Memory_changelong(Memory_Noupdate):
    def build_repre(self, input):
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        _, query = tf.nn.dynamic_rnn(cell, input[:, 768:, :], dtype=tf.float32)
        last = query
        mem_input = input[:, 768 - self.memory_maxlen:768, :]
        memory = self.build_memory(mem_input)
        print(memory)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        weights = []
        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query)
            query = tf.matmul(query, H) + read
            weights.append(weight)

        return tf.concat([query, last], axis=-1), weights

    def build_memory(self, input):
        memory = []
        k = int(math.log(self.memory_maxlen, 2))
        # with tf.variable_scope('GRU0'):
        #     cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        #     # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        #     outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        # memory.append(tf.expand_dims(state, axis=1))
        # outputs = tf.reshape(outputs, [-1, self.memory_maxlen / 3, 3, self.hidden_size])
        # input = tf.reshape(tf.gather(outputs, [1], axis=2),
        #                    [-1, self.memory_maxlen / 3, self.hidden_size])

        for i in range(8):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            outputs = tf.reshape(outputs, [-1, self.memory_maxlen / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2),
                               [-1, self.memory_maxlen / 2 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory


class Memory_inputq(Memory_Noupdate):
    def build_repre(self, input):
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        _, query = tf.nn.dynamic_rnn(cell, input[:, 768:, :], dtype=tf.float32)
        last = query
        query2 = input[:, -1, :]
        mem_input = input[:, 768 - self.memory_maxlen:768, :]
        memory = self.build_memory(mem_input)
        print(memory)
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        I = tf.get_variable('input_map', [self.input_dim * self.embedding_size, self.hidden_size])
        H2 = tf.get_variable('mapping2', [self.hidden_size, self.hidden_size])
        query2 = tf.matmul(query2, I)
        weights = []
        for _ in range(self.hops):
            # read, weight = self.attention(memory, memory, query)
            read2, weight = self.attention(memory, memory, query2)
            # query = tf.matmul(query, H) + read
            query2 = tf.matmul(query2, H2) + read2
            weights.append(weight)

        return tf.concat([query2, last], axis=-1), weights


class Memory_ind(Memory_taobao):
    def __init__(self, path, trainset, testset, input_dim, user_mem_maxlen, item_mem_maxlen, user_que_maxlen,
                 item_que_maxlen,
                 feature_size, user, item, user_feature_number, item_feature_number, learning_rate,
                 hidden_size,
                 embedding_size, batchsize, beta, hops, startpoint, emb_initializer=None, layers=6):
        self.emb_initializer = emb_initializer
        self.layers = layers
        Memory_taobao.__init__(self, path, trainset, testset, input_dim, user_mem_maxlen, item_mem_maxlen,
                               user_que_maxlen,
                               item_que_maxlen,
                               feature_size, user, item, user_feature_number, item_feature_number, learning_rate,
                               hidden_size,
                               embedding_size, batchsize, beta, hops, startpoint)

    def build_user_memory(self, input):
        memory = []
        with tf.variable_scope('GRU0'):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        memory.append(tf.expand_dims(state, axis=1))
        outputs = tf.reshape(outputs, [-1, self.user_mem_maxlen / 3, 3, self.hidden_size])
        input = tf.reshape(tf.gather(outputs, [2], axis=2),
                           [-1, self.user_mem_maxlen / 3, self.hidden_size])

        for i in range(5):
            with tf.variable_scope('GRU%s' % (i + 1)):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            outputs = tf.reshape(outputs, [-1, self.user_mem_maxlen / 3 / 4 ** (i + 1), 4, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [3], axis=2),
                               [-1, self.user_mem_maxlen / 3 / 4 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_item_memory(self, input):
        memory = []
        with tf.variable_scope('GRU0'):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        memory.append(tf.expand_dims(state, axis=1))
        outputs = tf.reshape(outputs, [-1, self.item_mem_maxlen / 5, 5, self.hidden_size])
        input = tf.reshape(tf.gather(outputs, [4], axis=2),
                           [-1, self.item_mem_maxlen / 5, self.hidden_size])

        for i in range(6):
            with tf.variable_scope('GRU%s' % (i + 1)):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            outputs = tf.reshape(outputs, [-1, self.item_mem_maxlen / 5 / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2),
                               [-1, self.item_mem_maxlen / 5 / 2 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_user_repre(self):
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        _, query = tf.nn.dynamic_rnn(cell, self.que_uin, dtype=tf.float32, sequence_length=self.user_que_len)
        last = query
        memory = self.build_user_memory(self.mem_uin)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])

        def cond(query, H, memory, hop, hops):
            return hop < self.hops

        def one_hop(query, H, memory, hop, hops):
            read, score = self.attention(memory, memory, query, 6)
            query = tf.matmul(query, H) + read
            hop += 1
            return query, H, memory, hop, hops

        query, H, memory, hop, self.hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, self.hops])

        return tf.concat([query, last], axis=-1)

    def build_item_repre(self):
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        _, query = tf.nn.dynamic_rnn(cell, self.que_iin, dtype=tf.float32, sequence_length=self.item_que_len)
        last = query
        memory = self.build_item_memory(self.mem_iin)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])

        def cond(query, H, memory, hop, hops):
            return hop < self.hops

        def one_hop(query, H, memory, hop, hops):
            read, score = self.attention(memory, memory, query, 7)
            query = tf.matmul(query, H) + read
            hop += 1
            return query, H, memory, hop, hops

        query, H, memory, hop, self.hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, self.hops])

        return tf.concat([query, last], axis=-1)

    def embedding(self):
        if self.emb_initializer is not None:
            emb_mtx = tf.get_variable('emb_mtx', initializer=self.emb_initializer)
        else:
            emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])

        mem_uin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.user_mem),
                             [-1, self.user_mem_maxlen, self.user_feature_number * self.embedding_size])
        que_uin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.user_que),
                             [-1, self.user_que_maxlen, self.user_feature_number * self.embedding_size])
        mem_iin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.item_mem),
                             [-1, self.item_mem_maxlen, self.item_feature_number * self.embedding_size])
        que_iin = tf.reshape(tf.nn.embedding_lookup(emb_mtx, self.item_que),
                             [-1, self.item_que_maxlen, self.item_feature_number * self.embedding_size])

        return mem_uin, que_uin, mem_iin, que_iin

    def train(self, epochs=1):
        step = 0
        flag = True
        losses = []
        for epoch in range(epochs):
            if not flag:
                break
            print('---------------epoch %s---------------' % epoch)
            start = time.time()
            for _, data in DataLoader_crop(self.trainset, self.batchsize):
                step += 1
                feed_dict = {
                    self.user_mem: data[1],
                    self.user_que: data[2],
                    self.item_mem: data[5],
                    self.item_que: data[6],
                    self.user_mem_len: data[3],
                    self.user_que_len: data[4],
                    self.item_mem_len: data[7],
                    self.item_que_len: data[8],
                    self.label: data[0],
                    self.keep_prob: 0.5,
                }
                _, loss = self.sess.run(fetches=[self.train_step, self.cross_entropy],
                                        feed_dict=feed_dict)
                if step % 10 == 0:
                    losses.append(self.eval(step, self.testset, 'test'))
                    if len(losses) >= 3 and losses[-1] < losses[-2] < losses[-3]:
                        flag = False
                        break
                if step % 40 == 0:
                    self.eval(step, self.trainset, 'train')
                if len(losses) >= 3 and losses[-1] > losses[-2] and losses[-1] > losses[-3]:
                    self.save_model()
        if flag:
            self.save_model()

    def eval(self, step, dataset, prefix):
        labels = []
        preds = []
        losses = []
        for _, data in DataLoader_crop(dataset, self.batchsize):
            labels += data[0]
            feed_dict = {
                self.user_mem: data[1],
                self.user_que: data[2],
                self.item_mem: data[5],
                self.item_que: data[6],
                self.user_mem_len: data[3],
                self.user_que_len: data[4],
                self.item_mem_len: data[7],
                self.item_que_len: data[8],
                self.label: data[0],
                self.keep_prob: 0.5,
            }
            loss, pred = self.sess.run(fetches=[self.log_loss, self.prediction],
                                       feed_dict=feed_dict)
            losses.append(loss)
            preds += pred.tolist()
        testloss = np.average(losses)
        testauc = roc_auc_score(labels, preds)
        print("%s\tSTEP: %s\tLOSS: %.4f\tAUC: %.4f" % (prefix, step, testloss, testauc))
        result = [testloss, testauc]
        self.log(step, result, prefix)
        return testauc


class Memory_noshort(Memory_Noupdate):
    def memory_loss(self, memory, k):
        memory = tf.split(memory, k, axis=1)
        loss = 0
        for i in range(k):
            for j in range(i, k):
                loss += tf.reduce_sum(memory[i] * memory[j])
        return loss

    def build_memory(self, input):
        memory = []
        for i in range(self.layers):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            outputs = tf.reshape(outputs, [-1, 1024 / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2), [-1, 1024 / 2 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_repre(self, input):
        zeros = tf.zeros_like(input[:, :23, :], dtype=tf.float32)
        input = tf.concat([zeros, input], axis=1)
        memory = self.build_memory(input)
        loss = self.memory_loss(memory, self.layers)
        query = input[:, -1, :]
        last = query
        I = tf.get_variable('input_map', [self.input_dim * self.embedding_size, self.hidden_size])
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        query = tf.matmul(query, I)
        weights = []

        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query)
            query = tf.matmul(query, H) + read
            weights.append(weight)

        return tf.concat([query, last], axis=-1), loss

    def build_graph(self):
        with tf.variable_scope('Embedding'):
            if self.emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=self.emb_initializer)
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [self.feature_size, self.embedding_size])

            input = tf.reshape(
                tf.nn.embedding_lookup(self.emb_mtx, self.sequence),
                [-1, self.maxlen, self.input_dim * self.embedding_size])

        with tf.variable_scope('repre'):
            repre, self.memory_loss = self.build_repre(input)

        self.build_fc_net(repre, reg=True)

class Memory_noshort_dual(Memory_ind):
    # def __init__(self, path, trainset, testset, input_dim, user_mem_maxlen, item_mem_maxlen, user_que_maxlen,
    #              item_que_maxlen,
    #              feature_size, user, item, user_feature_number, item_feature_number, learning_rate,
    #              hidden_size,
    #              embedding_size, batchsize, beta, hops, startpoint, emb_initializer=None, layers=6):
    #     self.emb_initializer = emb_initializer
    #     self.layers= layers
    #     Memory_taobao.__init__(self, path, trainset, testset, input_dim, user_mem_maxlen, item_mem_maxlen,
    #                            user_que_maxlen,
    #                            item_que_maxlen,
    #                            feature_size, user, item, user_feature_number, item_feature_number, learning_rate,
    #                            hidden_size,
    #                            embedding_size, batchsize, beta, hops, startpoint)

    def build_user_memory(self, input):
        memory = []

        for i in range(self.layers):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            outputs = tf.reshape(outputs, [-1, 1024 / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2),
                               [-1, 1024 / 2 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_item_memory(self, input):
        memory = []
        with tf.variable_scope('GRU0'):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        memory.append(tf.expand_dims(state, axis=1))
        outputs = tf.reshape(outputs, [-1, 192 / 3, 3, self.hidden_size])
        input = tf.reshape(tf.gather(outputs, [2], axis=2),
                           [-1, 192 / 3, self.hidden_size])

        for i in range(7):
            with tf.variable_scope('GRU%s' % (i + 1)):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            outputs = tf.reshape(outputs, [-1, 192 / 3 / 2 ** (i + 1), 2, self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [1], axis=2),
                               [-1, 192 / 3 / 2 ** (i + 1), self.hidden_size])

        memory = tf.concat(memory, axis=1)
        return memory

    def build_user_repre(self):
        input = tf.concat([self.mem_uin, self.que_uin], axis=1)
        # cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        # _, query = tf.nn.dynamic_rnn(cell, self.que_uin, dtype=tf.float32, sequence_length=self.user_que_len)
        # last = query
        query = input[:, -1, :]
        last = query

        input = tf.concat([tf.zeros_like(input[:, :1024 - 1001, :], dtype=tf.float32), input], axis=1)
        memory = self.build_user_memory(input)
        loss = self.memory_loss(memory, self.layers)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        M = tf.get_variable('input_map', [self.user_feature_number * self.embedding_size, self.hidden_size])
        query = tf.matmul(query, M)

        def cond(query, H, memory, hop, hops):
            return hop < self.hops

        def one_hop(query, H, memory, hop, hops):
            read, score = self.attention(memory, memory, query, 6)
            query = tf.matmul(query, H) + read
            hop += 1
            return query, H, memory, hop, hops

        query, H, memory, hop, self.hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, self.hops])

        return tf.concat([query, last], axis=-1), loss

    def build_item_repre(self):
        # cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        # _, query = tf.nn.dynamic_rnn(cell, self.que_iin, dtype=tf.float32, sequence_length=self.item_que_len)
        # last = query
        input = tf.concat([self.mem_iin, self.que_iin], axis=1)
        query = input[:, -1, :]
        last = query
        input = tf.concat([tf.zeros_like(input[:, :192 - 184, :], dtype=tf.float32), input], axis=1)
        memory = self.build_item_memory(input)
        loss = self.memory_loss(memory, 8)
        hop = 0
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        M = tf.get_variable('input_map', [self.item_feature_number * self.embedding_size, self.hidden_size])
        query = tf.matmul(query, M)

        def cond(query, H, memory, hop, hops):
            return hop < self.hops

        def one_hop(query, H, memory, hop, hops):
            read, score = self.attention(memory, memory, query, 7)
            query = tf.matmul(query, H) + read
            hop += 1
            return query, H, memory, hop, hops

        query, H, memory, hop, self.hops = tf.while_loop(cond, one_hop, [query, H, memory, hop, self.hops])

        return tf.concat([query, last], axis=-1), loss


class amazon_noshort(taobao_noshort):
    def build_user_memory(self, input):
        memory = []
        lengths = [2, 2, 5, 5, 1]
        len = self.item_maxlen
        for i in range(5):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            # memory.append(state[3])\
            len /= lengths[i]
            outputs = tf.reshape(outputs, [-1, len, lengths[i], self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [lengths[i] - 1], axis=2),
                               [-1, len, self.hidden_size])
        memory = tf.concat(memory, axis=1)
        return memory

    def build_item_memory(self, input):
        memory = []
        lengths = [2, 2, 5, 5, 1]
        len = self.item_maxlen
        for i in range(5):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            memory.append(tf.expand_dims(state, axis=1))
            # memory.append(state[3])
            len /= lengths[i]
            outputs = tf.reshape(outputs, [-1, len, lengths[i], self.hidden_size])
            input = tf.reshape(tf.gather(outputs, [lengths[i] - 1], axis=2),
                               [-1, len, self.hidden_size])
        memory = tf.concat(memory, axis=1)
        return memory

    def build_user_repre(self):
        last = self.uin[:, -1, :]
        query = last
        memory = self.build_user_memory(self.uin)
        I = tf.get_variable('input_map', [self.user_feature_number * self.embedding_size, self.hidden_size])
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        query = tf.matmul(query, I)
        weights = []
        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query, 6)
            query = tf.matmul(query, H) + read
            weights.append(weight)

        return tf.concat([query, last], axis=-1), weights

    def build_item_repre(self):
        index = tf.range(0, tf.shape(self.iin)[0]) * self.item_maxlen + self.item_len - 1
        last = self.iin[:, -1, :]
        query = last
        memory = self.build_item_memory(self.iin)
        I = tf.get_variable('input_map', [self.item_feature_number * self.embedding_size, self.hidden_size])
        H = tf.get_variable('mapping', [self.hidden_size, self.hidden_size])
        query = tf.matmul(query, I)
        weights = []
        for _ in range(self.hops):
            read, weight = self.attention(memory, memory, query, 5)
            query = tf.matmul(query, H) + read
            weights.append(weight)

        return tf.concat([query, last], axis=-1), weights


if __name__ == '__main__':
    # fin = open('../../data/taobao/dataset_fp.pkl', 'rb')
    # trainset = pkl.load(fin)
    # testset = pkl.load(fin)
    # feature_size = pkl.load(fin)
    # fin.close()
    #
    # model = taobao_noshort('../model/memory_taobao_noshort/', trainset, testset, 1, feature_size=feature_size,
    #                        learning_rate=0.003, batchsize=128, hidden_size=32, embedding_size=20, beta=0, hops=5,
    #                        user=True, item=False, item_maxlen=35, user_maxlen=300, startpoint=0, user_feature_number=4,
    #                        item_feature_number=3)
    # model.train(6)

    # fin = open('../../data/amazon/Electronics/dataset_fp.pkl', 'rb')
    # trainset = pkl.load(fin)
    # testset = pkl.load(fin)
    # feature_size = pkl.load(fin)
    # fin.close()
    #
    # model = amazon_noshort('../model/memory_amazon_noshort/', trainset, testset, 1, feature_size=feature_size,
    #                        learning_rate=0.001, batchsize=128, hidden_size=32, embedding_size=18, beta=0, hops=5,
    #                        user=True, item=False, item_maxlen=100, user_maxlen=100, startpoint=0, user_feature_number=3,
    #                        item_feature_number=2)
    # model.train(6)

    # fin = open('../../data/amazon/Electronics/dataset_crop.pkl', 'rb')
    # trainset = pkl.load(fin)
    # testset = pkl.load(fin)
    # feature_size = pkl.load(fin)
    # fin.close()
    # paras = []
    # for lr in [1e-3, 5e-3, 5e-4]:
    #     for hidden_size in [32, 16]:
    #         for embedding_size in [16]:
    #             for beta in [0]:
    #                 for hop in [1, 3, 5, 7]:
    #                     for batchsize in [64, 128, 256]:
    #                         paras.append((lr, hidden_size, embedding_size, beta, hop, batchsize))
    # random.shuffle(paras)
    # for i in range(len(paras)):
    #     print(i)
    #     print(paras[i])
    #     lr, hidden_size, embedding_size, beta, hop, batchsize = paras[i]
    #     path = '../model/amazon_inputq/%s_%s_%s_%s_%s_%s' % (
    #         str(lr), hidden_size, embedding_size, str(beta), hop, batchsize)
    #     if os.path.exists(path):
    #         continue
    #     else:
    #         os.makedirs(path)
    #     model = Memory_amazon(path, trainset, testset, 1, feature_size=feature_size,
    #                           learning_rate=lr, batchsize=batchsize, hidden_size=hidden_size,
    #                           embedding_size=embedding_size, beta=beta, hops=hop,
    #                           user=True, item=False, item_que_maxlen=10, item_mem_maxlen=90, user_que_maxlen=10,
    #                           user_mem_maxlen=90, startpoint=0,
    #                           user_feature_number=3,
    #                           item_feature_number=2)
    #     model.train(6)
    # for i in range(len(paras)):
    #     print(i)
    #     print(paras[i])
    #     lr, hidden_size, embedding_size, beta, hop, batchsize = paras[i]
    #     path = '../model/taobao_noshort/%s_%s_%s_%s_%s_%s' % (
    #         str(lr), hidden_size, embedding_size, str(beta), hop, batchsize)
    #     if os.path.exists(path):
    #         continue
    #     else:
    #         os.makedirs(path)
    #         model = taobao_noshort('../model/memory_taobao_noshort/', trainset, testset, 1, feature_size=feature_size,
    #                                learning_rate=lr, batchsize=batchsize, hidden_size=hidden_size, embedding_size=embedding_size, beta=0, hops=hop,
    #                                user=True, item=False, item_maxlen=36, user_maxlen=300, startpoint=0, user_feature_number=4,
    #                                item_feature_number=3)
    #         model.train(6)

#    fin = open('../../data/taobao/dataset_crop.pkl', 'rb')
#    trainset = pkl.load(fin)
#    testset = pkl.load(fin)
#    feature_size = pkl.load(fin)
#    fin.close()
#    paras = []
#    for lr in [1e-3, 5e-3, 5e-4]:
#        for hidden_size in [32, 16]:
#            for embedding_size in [16]:
#                for beta in [0]:
#                    for hop in [3, 5, 7]:
#                        for batchsize in [128, 256, 512]:
#                                paras.append((lr, hidden_size, embedding_size, beta, hop, batchsize))
#    random.shuffle(paras)
#    for i in range(len(paras)):
#        print(i)
#        print(paras[i])
#        lr, hidden_size, embedding_size, beta, hop, batchsize= paras[i]
#        path = '../model/taobao_inputq/%s_%s_%s_%s_%s_%s' % (
#            str(lr), hidden_size, embedding_size, str(beta), hop, batchsize)
#        if os.path.exists(path):
#            continue
#        else:
#            os.makedirs(path)
#        model = Memory_taobao(path, trainset, testset, 1, feature_size=feature_size,
#                              learning_rate=lr, batchsize=batchsize, hidden_size=hidden_size, embedding_size=embedding_size, beta=beta, hops=hop,
#                              user=True, item=False, item_mem_maxlen=27, user_mem_maxlen=256, item_que_maxlen=8,
#                              user_que_maxlen=44, startpoint=0, user_feature_number=4, item_feature_number=3)
#        model.train(6)
    #
    # fin = open('../../data/amazon/Electronics/dataset_crop.pkl', 'rb')
    # trainset = pkl.load(fin)
    # testset = pkl.load(fin)
    # feature_size = pkl.load(fin)
    # fin.close()
    # model = Memory_amazon('../model/memory_amazon/', trainset, testset, 1, feature_size=feature_size,
    #                       learning_rate=0.001, batchsize=2000, hidden_size=32, embedding_size=20, beta=0, hops=3,
    #                       user=True, item=True, item_mem_maxlen=90, user_mem_maxlen=90, item_que_maxlen=10,
    #                       user_que_maxlen=10, startpoint=0, user_feature_number=3, item_feature_number=2)
    # model.train(6)

# fin = open('../../data/amazon/Electronics/dataset_crop.pkl', 'rb')
# trainset = pkl.load(fin)
# testset = pkl.load(fin)
# feature_size = pkl.load(fin)
# fin.close()
# model = srnn_dmem('../model/srnn_dmem/final', trainset, testset, input_dim=1, user_mem_maxlen=90, item_mem_maxlen=90, user_que_maxlen=10,
#              item_que_maxlen=10,
#              feature_size=feature_size, user=True, item=True, user_feature_number=3, item_feature_number=2, learning_rate=0.001,
#              hidden_size=32,
#              embedding_size=20, batchsize=32, beta=0, hops=3, startpoint=2000)
# model.train(6)
# model.get_weight()

###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

    train_set = '/home/weijie.bwj/UIC/data/cvr_dataset/train_corpus_total.txt'
    test_set = '/home/weijie.bwj/UIC/data/cvr_dataset/test_corpus_total.txt'
    user_feature_size = len(pkl.load(open('/home/weijie.bwj/UIC/data/cvr_dataset/nick_dic.pkl')))
    feature_size = np.load('/home/weijie.bwj/UIC/data/cvr_dataset/graph_emb.npy').shape[0] + user_feature_size
    max_len = 256 + 1
    item_part_fnum = 2
    user_part_fnum = 2
    emb_initializer = np.concatenate((np.load('/home/weijie.bwj/UIC/data/cvr_dataset/graph_emb.npy'),
                                      np.zeros(
                                          [len(pkl.load(open('/home/weijie.bwj/UIC/data/cvr_dataset/nick_dic.pkl'))),
                                           16])),
                                     0).astype(np.float32)

# model = Memory_inputq('../model/inputq/', train_set, test_set, 2, 1001, 768, feature_size,
#                         user_feature_size,
#                         emb_initializer, learning_rate=0.005, hidden_size=32,
#                         embedding_size=16, batchsize=500,
#                         beta=0, hop=9, layers=5)
# model.train(5)

# model.get_weights()

# for mem_maxlen in [512]:
#     model = Memory_changelong('../model/memory_cl_2/%s' % mem_maxlen, train_set, test_set, 2, 1001, mem_maxlen,
#                               feature_size,
#                               user_feature_size,
#                               emb_initializer, learning_rate=0.005, hidden_size=32,
#                               embedding_size=16, batchsize=512,
#                               beta=0, hop=6)
#     model.train(5)

    paras = []
    for lr in [5e-3, 3e-3, 4e-3, 2e-3, 1e-3]:
        for hidden_size in [32]:
            for embedding_size in [16]:
                for beta in [0]:
                    for hop in [1, 3, 5, 7, 9]:
                        for batchsize in [300, 500, 600, 700]:
                            for layers in [5, 6, 7, 8, 9, 10, 11]:
                                paras.append((lr, hidden_size, embedding_size, beta, hop, batchsize, layers))
    random.shuffle(paras)
    for i in range(len(paras)):
        print(i)
        print(paras[i])
        lr, hidden_size, embedding_size, beta, hop, batchsize, layers = paras[i]
        path = '../model/noshort_reg/%s_%s_%s_%s_%s_%s_%s' % (
            str(lr), hidden_size, embedding_size, str(beta), hop, batchsize, layers)
        if os.path.exists(path):
            continue
        else:
            os.makedirs(path)
        model = Memory_noshort(path, train_set, test_set, 2, 1001, 768, feature_size,
                              user_feature_size,
                              emb_initializer, learning_rate=lr, hidden_size=hidden_size,
                              embedding_size=embedding_size, batchsize=batchsize,
                             beta=beta, hop=hop, layers=layers)
        model.train(epochs=5)

# train_set = '/home/weijie.bwj/UIC/data/cvr_dataset/train_corpus_total_dual.txt'
# test_set = '/home/weijie.bwj/UIC/data/cvr_dataset/test_corpus_total_dual.txt'
# pv_cnt = 19002
# feature_size = pv_cnt + np.load('/home/weijie.bwj/UIC/data/cvr_dataset/graph_emb.npy').shape[0] + len(
#     pkl.load(open('/home/weijie.bwj/UIC/data/cvr_dataset/nick_dic.pkl')))
# max_len_item = 1000 + 1
# max_len_user = 184
# item_part_fnum = 2
# user_part_fnum = 1
# emb_initializer = np.concatenate((np.load('/home/weijie.bwj/UIC/data/cvr_dataset/graph_emb.npy'), np.zeros(
#     [len(pkl.load(open('/home/weijie.bwj/UIC/data/cvr_dataset/nick_dic.pkl'))), 16]), np.zeros([pv_cnt, 16])),
#                                  0).astype(np.float32)
#
# model = Memory_ind('../model/memory_ind_4/', train_set, test_set, 1, feature_size=feature_size,
#                    learning_rate=0.003, batchsize=512, hidden_size=32, embedding_size=16, beta=0, hops=5,
#                    user=True, item=False, item_mem_maxlen=160, user_mem_maxlen=768, item_que_maxlen=24,
#                    user_que_maxlen=233, startpoint=0, user_feature_number=item_part_fnum,
#                    item_feature_number=user_part_fnum, emb_initializer=emb_initializer)
# model.train(6)
# paras = []
# for lr in [5e-3, 6e-3, 4e-3]:
#     for hidden_size in [32, 16]:
#         for embedding_size in [16]:
#             for beta in [0]:
#                 for hop in [1, 2, 3, 4, 5]:
#                     for batchsize in [512, 256, 1024]:
#                         for layers in [5, 6, 7, 8, 9, 10, 11]:
#                             paras.append((lr, hidden_size, embedding_size, beta, hop, batchsize, layers))
# random.shuffle(paras)
# for i in range(len(paras)):
#     print(i)
#     print(paras[i])
#     lr, hidden_size, embedding_size, beta, hop, batchsize, layers = paras[i]
#     path = '../model/noshort_dual/%s_%s_%s_%s_%s_%s_%s' % (
#         str(lr), hidden_size, embedding_size, str(beta), hop, batchsize, layers)
#     if os.path.exists(path):
#         continue
#     else:
#         os.makedirs(path)
#     model = Memory_noshort_dual(path, train_set, test_set, 1, feature_size=feature_size,
#                        learning_rate=lr, batchsize=batchsize, hidden_size=hidden_size, embedding_size=16, beta=beta,
#                        hops=hop,
#                        user=True, item=True, item_mem_maxlen=160, user_mem_maxlen=768, item_que_maxlen=24,
#                        user_que_maxlen=233, startpoint=0, user_feature_number=2, item_feature_number=1,
#                        emb_initializer=emb_initializer, layers=layers)
#     model.train(epochs=5)