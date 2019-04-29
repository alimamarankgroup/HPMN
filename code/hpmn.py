import cPickle as pkl
import os
import random
import shutil
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

from data_loader import DataLoader, DataLoader_Mul, DataLoader_crop, CroppedLoader

random.seed(42)


class Hpmn_Basic(object):
    def __init__(self,
                 path,
                 trainset,
                 testset,
                 feature_size,
                 user_dim,
                 item_dim,
                 learning_rate,
                 hidden_size,
                 embedding_size,
                 hop,
                 user_layers,
                 item_layers,
                 user_num_layers,
                 item_num_layers,
                 user,
                 item,
                 emb_initializer=None,
                 l2_reg=0,
                 memory_reg=1e-5):
        self.graph = tf.Graph()
        self._path = path
        self.trainset = trainset
        self.testset = testset
        self._save_path, self._logs_path = None, None
        self.feature_size = feature_size
        self.cross_entropy, self.train_step, self.prediction, self.memory_loss = None, None, None, None
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.memory_reg = memory_reg
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.emb_initializer = emb_initializer
        self.hop = hop
        self.keep_prob = None
        self.label = None
        self.user_layers, self.item_layers = user_layers, item_layers
        self.user_num_layers, self.item_num_layers = user_num_layers, item_num_layers
        self.user_dim, self.item_dim = user_dim, item_dim
        self.user, self.item = user, item
        with self.graph.as_default():
            self.define_inputs()
            self.build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/ckpt' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

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
            self.summary_writer = logswriter(
                self._path + '/logs', graph=self.graph)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def log(self, step, result):
        print(
            'Step: %s\tTrain AUC: %.5f\tTrain Loss: %.5f\tTrain Mem_loss: %.5f\
            \tTest AUC: %.5f\tTest Loss: %.5f\tTest Mem_loss: %.5f' %
            (str(step), result[0], result[1], result[2], result[3], result[4],
             result[5]))
        fout = open(self._path + '/result.log', 'a')
        fout.write('%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' %
                   (str(step), result[0], result[1], result[2], result[3],
                    result[4], result[5]))

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError(
                'Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

    def build_memory(self, inp, li_layer, maxlen, num_layer):
        memory = []
        assert num_layer <= len(li_layer)
        for i in range(num_layer):
            with tf.variable_scope('GRU%s' % i):
                cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
                outputs, states = tf.nn.dynamic_rnn(
                    cell, inp, dtype=tf.float32)
            memory.append(tf.expand_dims(states, axis=1))
            maxlen /= li_layer[i]
            maxlen = int(maxlen)
            outputs = tf.reshape(outputs,
                                 [-1, maxlen, li_layer[i], self.hidden_size])
            inp = tf.reshape(
                tf.gather(outputs, [li_layer[i] - 1], axis=2),
                [-1, maxlen, self.hidden_size])
        memory = tf.concat(memory, axis=1)
        loss = self.get_covreg(memory, num_layer)
        return memory, loss

    def attention(self, key, value, query, k):
        k = key.get_shape().as_list()[1]
        queries = tf.tile(tf.expand_dims(query, 1), [1, k, 1])  # [B, T, Dk]
        inp = tf.concat([queries, key, queries - key, queries * key], axis=-1)
        fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None)  # [B, T, 1]

        score = tf.nn.softmax(tf.reshape(fc3, [-1, k]))  # [B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2))
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return atten_output_sum, score

    def get_reg(self, memory, k):
        """get_reg

        :param memory:
        :param k:
        """
        memory = tf.split(memory, k, axis=1)
        loss = 0
        for i in range(k):
            for j in range(i, k):
                loss += tf.reduce_sum(memory[i] * memory[j])
        return loss / (k * (k - 1))

    def get_covreg(self, memory, k):
        mean = tf.reduce_mean(memory, axis=2, keep_dims=True)
        C = memory - mean
        C = tf.matmul(C, tf.transpose(C, [0, 2, 1])) / tf.cast(
            tf.shape(memory)[2], tf.float32)
        C_diag = tf.linalg.diag_part(C)
        C_diag = tf.linalg.diag(C_diag)
        C = C - C_diag
        norm = tf.norm(C, ord='fro', axis=[1, 2])
        return tf.reduce_sum(norm)

    def query_memory(self, query, memory, k):
        query = tf.layers.dense(query, self.hidden_size)
        H = tf.get_variable('map', [self.hidden_size, self.hidden_size])
        weights = []

        for _ in range(self.hop):
            read, weight = self.attention(memory, memory, query, k)
            query = tf.matmul(query, H) + read
            weights.append(weight)

        return query, weights[0]

    def build_fc_net(self, inp, reg=False):
        """build_fc_net

        :param inp:
        :param reg:
        """
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.elu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.elu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=tf.nn.sigmoid, name='fc3')
        # output
        self.prediction = tf.reshape(fc3, [
            -1,
        ])

        # loss
        self.log_loss = tf.losses.log_loss(self.label, self.prediction)
        self.cross_entropy = self.log_loss
        for v in tf.trainable_variables():
            self.cross_entropy += self.l2_reg * tf.nn.l2_loss(v)
        if reg:
            self.cross_entropy += self.memory_reg * self.memory_loss
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        gvs = self.optimizer.compute_gradients(self.cross_entropy)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                      for grad, var in gvs]
        self.train_step = self.optimizer.apply_gradients(capped_gvs)


class Hpmn_Industry(Hpmn_Basic):
    def __init__(self,
                 path,
                 trainset,
                 testset,
                 feature_size,
                 user_dim,
                 item_dim,
                 user_maxlen,
                 item_maxlen,
                 learning_rate,
                 hidden_size,
                 embedding_size,
                 hop,
                 user_layers,
                 item_layers,
                 user_num_layers,
                 item_num_layers,
                 user,
                 item,
                 emb_initializer=None,
                 l2_reg=0,
                 memory_reg=1e-5):
        self.user_maxlen, self.item_maxlen = user_maxlen, item_maxlen
        super(Hpmn_Industry, self).__init__(
            path, trainset, testset, feature_size, user_dim, item_dim,
            learning_rate, hidden_size, embedding_size, hop, user_layers,
            item_layers, user_num_layers, item_num_layers, user, item,
            emb_initializer, l2_reg, memory_reg)

    def define_inputs(self):
        self.user_inp = tf.placeholder(
            tf.int32, shape=[None, self.user_maxlen, self.user_dim])
        self.item_inp = tf.placeholder(
            tf.int32, shape=[None, self.item_maxlen, self.item_dim])
        self.user_len = tf.placeholder(
            tf.int32, shape=[
                None,
            ])
        self.item_len = tf.placeholder(
            tf.int32, shape=[
                None,
            ])
        self.label = tf.placeholder(
            tf.int32, shape=[
                None,
            ])
        self.keep_prob = tf.placeholder(tf.float32)

    def embedding(self):
        if self.emb_initializer is not None:
            emb_mtx = tf.get_variable(
                'emb_mtx', initializer=self.emb_initializer)
        else:
            emb_mtx = tf.get_variable('emb_mtx',
                                      [self.feature_size, self.embedding_size])

        uinp = tf.reshape(
            tf.nn.embedding_lookup(emb_mtx, self.user_inp),
            [-1, self.user_maxlen, self.user_dim * self.embedding_size])

        iinp = tf.reshape(
            tf.nn.embedding_lookup(emb_mtx, self.item_inp),
            [-1, self.item_maxlen, self.item_dim * self.embedding_size])

        return uinp, iinp

    def build_graph(self):
        with tf.variable_scope("Embedding"):
            uinp, iinp = self.embedding()
        with tf.variable_scope('User'):
            zeros = tf.zeros_like(uinp[:, :23, :], dtype=tf.float32)
            uinp = tf.concat([zeros, uinp], axis=1)
            memory, umloss = self.build_memory(uinp, self.user_layers, 1024,
                                               self.user_num_layers)
            last = uinp[:, -2, :]
            query, self.user_weights = self.query_memory(
                last, memory, self.user_num_layers)
            user_repre = tf.concat([query, last], axis=-1)

        with tf.variable_scope('Item'):
            zeros = tf.zeros_like(iinp[:, :192 - 184, :], dtype=tf.float32)
            iinp = tf.concat([zeros, iinp], axis=1)
            memory, imloss = self.build_memory(iinp, self.item_layers, 192,
                                               self.item_num_layers)
            last = iinp[:, -1, :]
            query, self.item_weights = self.query_memory(
                last, memory, self.item_num_layers)
            item_repre = tf.concat([query, last], axis=-1)

        with tf.variable_scope('concat'):
            if self.user:
                if self.item:
                    repre = tf.concat([user_repre, item_repre], axis=-1)
                    self.memory_loss = imloss + umloss
                else:
                    repre = user_repre
                    self.memory_loss = umloss
            else:
                repre = item_repre
                self.memory_loss = imloss

        with tf.variable_scope('output'):
            self.build_fc_net(repre, True)

    def train(self, epochs, batchsize):
        step = 0
        count = 0
        best = 0.
        for _ in range(epochs):
            for _, data in DataLoader_Mul(self.trainset, batchsize):
                feed_dict = {
                    self.label: data[0],
                    self.user_inp: data[1],
                    self.user_len: data[2],
                    self.item_inp: data[3],
                    self.item_len: data[4],
                    self.keep_prob: 0.5,
                }
                self.sess.run(fetches=[self.train_step], feed_dict=feed_dict)
                step += 1
                if step % 10 == 0:
                    result = list(self.eval(self.trainset, 4 * batchsize))
                    result += list(self.eval(self.testset, 4 * batchsize))
                    self.log(step, result)
                    if result[3] <= best:
                        count += 1
                        if count > 3:
                            return best
                    else:
                        count = 0
                        best = result[3]
        return best

    def eval(self, dataset, batchsize):
        labels = []
        preds = []
        mem_losses = []
        for _, data in DataLoader_Mul(dataset, batchsize):
            labels += data[0]
            feed_dict = {
                self.label: data[0],
                self.user_inp: data[1],
                self.user_len: data[2],
                self.item_inp: data[3],
                self.item_len: data[4],
                self.keep_prob: 1,
            }
            memory_loss, pred = self.sess.run(
                fetches=[self.memory_loss, self.prediction],
                feed_dict=feed_dict)
            mem_losses.append(memory_loss)
            preds += pred.tolist()
        mem_loss = np.average(mem_losses)
        auc = roc_auc_score(labels, preds)
        loss = log_loss(labels, preds)
        return auc, loss, mem_loss

    def get_weights(self):
        weights = []
        ids = []
        for _, data in DataLoader_Mul(self.trainset, 512):
            feed_dict = {
                self.label: data[0],
                self.user_inp: data[1],
                self.user_len: data[2],
                self.item_inp: data[3],
                self.item_len: data[4],
                self.keep_prob: 1,
            }
            user_weight, item_weight = self.sess.run(
                fetches=[self.user_weights, self.item_weights],
                feed_dict=feed_dict)
            weights += user_weight.tolist()
            ids += data[1][:, :, 1].tolist()
        for _, data in DataLoader_Mul(self.testset, 512):
            feed_dict = {
                self.label: data[0],
                self.user_inp: data[1],
                self.user_len: data[2],
                self.item_inp: data[3],
                self.item_len: data[4],
                self.keep_prob: 1,
            }
            user_weight, item_weight = self.sess.run(
                fetches=[self.user_weights, self.item_weights],
                feed_dict=feed_dict)
            weights += user_weight.tolist()
            ids += data[1][:, :, 1].tolist()

        weights = np.array(weights)
        ids = np.array(ids)
        np.save(self._path + '/weights_new.npy', weights)
        np.save(self._path + '/ids.npy', ids)


class Hpmn(Hpmn_Industry):
    def embedding(self):
        emb_mtx = tf.get_variable('emb_mtx',
                                  [self.feature_size, self.embedding_size])
        uid_mask_array = [[0.]] + [[1.]] * (self.feature_size - 1)
        mask_lookup_table = tf.get_variable(
            "mask_lookup_table", initializer=uid_mask_array, trainable=False)
        uinp = tf.reshape(
            tf.nn.embedding_lookup(emb_mtx, self.user_inp) *
            tf.nn.embedding_lookup(mask_lookup_table, self.user_inp),
            [-1, self.user_maxlen, self.user_dim * self.embedding_size])

        iinp = tf.reshape(
            tf.nn.embedding_lookup(emb_mtx, self.item_inp) *
            tf.nn.embedding_lookup(mask_lookup_table, self.item_inp),
            [-1, self.item_maxlen, self.item_dim * self.embedding_size])

        return uinp, iinp

    def build_graph(self):
        with tf.variable_scope("Embedding"):
            uinp, iinp = self.embedding()

        with tf.variable_scope("User"):
            memory, umloss = self.build_memory(
                uinp, self.user_layers, self.user_maxlen, self.user_num_layers)
            last = uinp[:, -1, :]
            query, self.user_weights = self.query_memory(
                last, memory, self.user_num_layers)
            user_repre = tf.concat([query, last], axis=-1)

        with tf.variable_scope("item"):
            memory, imloss = self.build_memory(
                iinp, self.item_layers, self.item_maxlen, self.item_num_layers)
            last = iinp[:, -1, :]
            query, self.item_weights = self.query_memory(
                last, memory, self.item_num_layers)
            item_repre = tf.concat([query, last], axis=-1)

        with tf.variable_scope('concat'):
            if self.user:
                if self.item:
                    repre = tf.concat([user_repre, item_repre], axis=-1)
                    self.memory_loss = imloss + umloss
                else:
                    repre = user_repre
                    self.memory_loss = umloss
            else:
                repre = item_repre
                self.memory_loss = imloss

        with tf.variable_scope('output'):
            self.build_fc_net(repre, True)

    def train(self, epochs, batchsize):
        step = 0
        count = 0
        best = 0.
        for _ in range(epochs):
            for _, data in DataLoader(self.trainset, batchsize):
                step += 1
                feed_dict = {
                    self.user_inp: data[1],
                    self.item_inp: data[3],
                    self.user_len: data[2],
                    self.item_len: data[4],
                    self.label: data[0],
                    self.keep_prob: 0.5,
                }
                self.sess.run(fetches=[self.train_step], feed_dict=feed_dict)
                if step % 100 == 0:
                    result = list(self.eval(self.trainset, 4 * batchsize))
                    result += list(self.eval(self.testset, 4 * batchsize))
                    self.log(step, result)
                    if result[3] <= best:
                        count += 1
                        if count > 3:
                            return best
                    else:
                        count = 0
                        best = result[3]

        return best

    def eval(self, dataset, batchsize):
        labels = []
        preds = []
        mem_losses = []
        for _, data in DataLoader(dataset, batchsize):
            labels += data[0]
            feed_dict = {
                self.user_inp: data[1],
                self.item_inp: data[3],
                self.user_len: data[2],
                self.item_len: data[4],
                self.label: data[0],
                self.keep_prob: 1.,
            }
            memory_loss, pred = self.sess.run(
                fetches=[self.memory_loss, self.prediction],
                feed_dict=feed_dict)
            mem_losses.append(memory_loss)
            preds += pred.tolist()
        mem_loss = np.average(mem_losses)
        auc = roc_auc_score(labels, preds)
        loss = log_loss(labels, preds)
        return auc, loss, mem_loss

    def get_weights(self):
        weights = []
        lengths = []
        labels = []
        for _, data in DataLoader(self.trainset, 512):
            feed_dict = {
                self.label: data[0],
                self.user_inp: data[1],
                self.user_len: data[2],
                self.item_inp: data[3],
                self.item_len: data[4],
                self.keep_prob: 1,
            }
            user_weight, item_weight = self.sess.run(
                fetches=[self.user_weights, self.item_weights],
                feed_dict=feed_dict)
            weights += user_weight.tolist()
            lengths += data[2]
            labels += data[0]
        for _, data in DataLoader(self.testset, 512):
            feed_dict = {
                self.label: data[0],
                self.user_inp: data[1],
                self.user_len: data[2],
                self.item_inp: data[3],
                self.item_len: data[4],
                self.keep_prob: 1,
            }
            user_weight, item_weight = self.sess.run(
                fetches=[self.user_weights, self.item_weights],
                feed_dict=feed_dict)
            weights += user_weight.tolist()
            lengths += data[2]
            labels += data[0]

        weights = np.array(weights)
        lengths = np.array(lengths)
        np.save(self._path + '/weights.npy', weights)
        np.save(self._path + '/lengths.npy', lengths)
        np.save(self._path + '/labels.npy', labels)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Useage: python hpmn.py [dataset]")
        exit(1)
    else:
        dataset_name = sys.argv[1]

    if dataset_name == 'amazon':
        fin = open('../data/amazon/dataset_hpmn.pkl', 'rb')
        trainset = pkl.load(fin)
        testset = pkl.load(fin)
        feature_size = pkl.load(fin)
        fin.close()
        model = Hpmn(
            'model/amazon/hpmn/',
            trainset,
            testset,
            feature_size,
            3,
            2,
            100,
            100,
            0.003,
            32,
            16,
            7, [2, 2, 5, 5, 1], [2, 2, 5, 5, 1],
            3,
            5,
            True,
            False,
            l2_reg=1e-5,
            memory_reg=5e-5)
        #model.train(2, 128)
        model.train(2, 10) #change the batch size to 128 when getting the full datasets
        model.save_model()

    elif dataset_name == 'taobao':
        fin = open('../data/taobao/dataset_hpmn.pkl', 'rb')
        trainset = pkl.load(fin)
        testset = pkl.load(fin)
        feature_size = pkl.load(fin)
        fin.close()
        model = Hpmn(
            'model/taobao/hpmn/',
            trainset,
            testset,
            feature_size,
            4,
            3,
            300,
            36,
            0.001,
            32,
            16,
            3, [2, 2, 3, 5, 5, 1], [2, 2, 3, 3, 1],
            4,
            5,
            True,
            False,
            l2_reg=0,
            memory_reg=1e-5)
        model.train(2, 128)
        model.save_model()
    else:
        print("Dataset must be one of taobao or amazon.")
        exit(1)
