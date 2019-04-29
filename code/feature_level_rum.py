import logging
import logging.config
import time

import Queue
import cPickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from data_loader import *
from sklearn.metrics import roc_auc_score
import os
import sys


class data_generation(object):
    def __init__(self, type, logger):
        print 'init'
        # np.random.seed(0)
        self.data_type = type
        self.logger = logger
        self.train_file_path = '../data/' + self.data_type + '_train_filtered'
        self.test_file_path = '../data/' + self.data_type + '_test_filtered'

        self.train_users = []
        self.train_items = []
        self.train_ratings = []
        self.train_labels = []

        self.test_users = []
        self.test_candidate_items = []
        self.test_batch_users = []
        self.test_batch_items = []
        self.test_batch_ratings = []
        self.test_batch_real_users = []
        self.test_batch_real_items = []
        self.test_batch_real_ratings = []

        self.neg_number = 1
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.user_number = 0
        self.item_number = 0
        self.records_number = 0

    def gen_train_data(self):
        self.data = pd.read_csv(self.train_file_path, names=['user', 'items'], dtype='str')
        is_first_line = 1
        for line in self.data.values:
            if is_first_line == 1:
                self.user_number = int(line[0]) + 1
                self.item_number = int(line[1]) + 1
                self.user_purchased_item = dict()
                self.user_purchased_item_number = np.zeros(self.user_number, dtype='int32')
                is_first_line = 0
            else:
                user_id = int(line[0])
                items_id = [i for i in line[1].split('@')]
                size = len(items_id)
                self.user_purchased_item_number[user_id] = size
                self.user_purchased_item[user_id] = [int(itm.split(':')[0]) for itm in items_id]
                for item in items_id:
                    itm = item.split(':')
                    self.train_users.append(int(user_id))
                    self.train_items.append(int(itm[0]))
                    self.train_ratings.append(int(float(itm[1])))
                    self.train_labels.append(1)
                    self.records_number += 1
                    for i in range(self.neg_number):
                        self.train_users.append(int(user_id))
                        neg = self.gen_neg(int(user_id))
                        self.train_items.append(neg)
                        self.train_ratings.append(0)
                        self.train_labels.append(0)
                        self.records_number += 1

    def gen_test_data(self):
        self.data = pd.read_csv(self.test_file_path, header=None, dtype='str')
        items = []
        is_first_line = 1
        for line in self.data.values:
            if is_first_line == 1:
                self.user_number = int(line[0]) + 1
                self.item_number = int(line[1]) + 1
                self.user_item_ground_true = dict()
                self.user_item_ground_true_number = np.zeros(self.user_number, dtype='int32')
                is_first_line = 0
            else:
                user_id = int(line[0])
                items_id = [i for i in line[1].split('@')]
                size = len(items_id)
                self.user_item_ground_true_number[user_id] = size
                self.user_item_ground_true[user_id] = [int(itm.split(':')[0]) for itm in items_id]

                self.test_batch_real_users.append(int(user_id))
                self.test_batch_real_items.append([int(itm.split(':')[0]) for itm in items_id])
                items += [int(itm.split(':')[0]) for itm in items_id]

        self.test_candidate_items = list(range(self.item_number))
        self.test_users = list(set(self.test_batch_real_users))

        for u in self.test_users:
            for item_id in self.test_candidate_items:
                self.test_batch_users.append(u)
                self.test_batch_items.append(item_id)

    def gen_neg(self, user_id):
        neg_item = np.random.randint(self.item_number)
        while neg_item in self.user_purchased_item[user_id]:
            neg_item = np.random.randint(self.item_number)
        return neg_item

    def shuffle(self):
        self.logger.info('shuffle ...')
        self.index = np.array(range(len(self.train_items)))
        np.random.shuffle(self.index)

        self.train_users = list(np.array(self.train_users)[self.index])
        self.train_items = list(np.array(self.train_items)[self.index])
        self.train_labels = list(np.array(self.train_labels)[self.index])
        self.train_ratings = list(np.array(self.train_ratings)[self.index])

    def gen_train_batch_data(self, batch_size):
        l = len(self.train_users)

        if self.train_batch_id + batch_size >= l:
            batch_users = self.train_users[self.train_batch_id:] + self.train_users[
                                                                   :self.train_batch_id + batch_size - l]
            batch_items = self.train_items[self.train_batch_id:] + self.train_items[
                                                                   :self.train_batch_id + batch_size - l]
            batch_ratings = self.train_ratings[self.train_batch_id:] + self.train_ratings[
                                                                       :self.train_batch_id + batch_size - l]
            batch_labels = self.train_labels[self.train_batch_id:] + self.train_labels[
                                                                     :self.train_batch_id + batch_size - l]
            # self.shuffle()
            self.train_batch_id = self.train_batch_id + batch_size - l
        else:
            batch_users = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
            batch_items = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
            batch_ratings = self.train_ratings[self.train_batch_id: self.train_batch_id + batch_size]
            batch_labels = self.train_labels[self.train_batch_id: self.train_batch_id + batch_size]

            self.train_batch_id = self.train_batch_id + batch_size

        # print batch_users[0:5], batch_items[0:5], batch_labels[0:5]
        return batch_users, batch_items, batch_ratings, batch_labels

    def gen_test_batch_data(self, user_number):
        l = len(self.test_users)
        if self.test_batch_id == len(self.test_candidate_items) * l:
            self.test_batch_id = 0

        batch_size = len(self.test_candidate_items) * user_number

        test_batch_users = self.test_batch_users[self.test_batch_id:self.test_batch_id + batch_size]
        test_batch_items = self.test_batch_items[self.test_batch_id:self.test_batch_id + batch_size]
        self.test_batch_id = self.test_batch_id + batch_size

        return test_batch_users, test_batch_items


class rum():
    def __init__(self, data_type):
        print 'init ...'
        # tf.set_random_seed(0)
        self.input_data_type = data_type
        logging.config.fileConfig('logging.conf')
        self.logger = logging.getLogger()

        # self.dg = data_generation(self.input_data_type, self.logger)
        # self.dg.gen_train_data()
        # self.dg.gen_test_data()
        #
        # self.train_user_purchsed_items_dict = self.dg.user_purchased_item
        # self.test_user_purchsed_items_dict = self.dg.user_item_ground_true
        #
        # self.user_number = self.dg.user_number
        # self.item_number = self.dg.item_number
        # self.neg_number = self.dg.neg_number
        # self.user_purchased_items = self.dg.user_purchased_item

        #################################################################
        self.user_number = 260000
        self.item_number = 260000
        print self.user_number
        print self.item_number
        #################################################################

        # self.test_users = self.dg.test_users
        # self.test_candidates = self.dg.test_candidate_items
        # self.test_real_u = self.dg.test_batch_real_users
        # self.test_real_i = self.dg.test_batch_real_items

        self.global_dimension = 40
        self.batch_size = 1
        self.memory_rows = 5
        self.recommend_new = 1
        self.K = 5
        self.results = []

        self.step = 0
        self.iteration = 10
        # self.display_step = self.dg.records_number

        self.initializer = tf.random_uniform_initializer(minval=0, maxval=0.1)
        self.c_init = tf.constant_initializer(value=0)

        self.len = tf.placeholder(tf.int32, shape=[], name='len')
        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')

        self.dual_user_id = tf.placeholder(tf.int32, shape=[None], name='dual_user_id')
        self.dual_item_id = tf.placeholder(tf.int32, shape=[None], name='dual_item_id')

        self.category_id = tf.placeholder(tf.int32, shape=[None], name='category_id')
        self.label = tf.placeholder(tf.float32, shape=[None], name='label')
        self.index = tf.placeholder(tf.float32, shape=[None], name='index')
        self.rating = tf.placeholder(tf.float32, shape=[None], name='rating')

        self.embedding_matrix = tf.get_variable('embedding_matrix', initializer=self.initializer,
                                                shape=[300000, self.global_dimension])
        # self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer, shape=[self.user_number, self.global_dimension])
        # self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer, shape=[self.item_number, self.global_dimension])
        self.user_embedding_matrix = self.embedding_matrix
        self.item_embedding_matrix = self.embedding_matrix

        self.rating_embedding_matrix = tf.get_variable('rating_embedding_matrix', initializer=self.initializer,
                                                       shape=[5, self.global_dimension])

        self.user_bias_vector = tf.get_variable('user_bias_vector', shape=[self.user_number])
        self.item_bias_vector = tf.get_variable('item_bias_vector', shape=[self.item_number])
        self.global_bias = tf.constant([0.0])

        self.feature_key = tf.get_variable('feature_key', shape=[self.memory_rows, self.global_dimension])
        self.memory = tf.get_variable('memory', shape=[self.user_number, self.memory_rows, self.global_dimension])
        self.dropout_keep_prob = 1.0

        self.dual_feature_key = tf.get_variable('dual_feature_key', shape=[self.memory_rows, self.global_dimension])
        self.dual_memory = tf.get_variable('dual_memory',
                                           shape=[self.user_number, self.memory_rows, self.global_dimension])

    def clear_memory(self):
        print 'clear memory'
        zeros = tf.constant(np.zeros((self.user_number, self.memory_rows, self.global_dimension)), dtype='float32')
        self.memory = tf.scatter_update(self.memory, range(self.user_number), zeros)

    def lookup_item_embedding(self):
        current_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        # current_catefory_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.category_id)
        # current_user_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.user_id)
        # emb = tf.concat([current_item_embedding, current_catefory_embedding], axis = 1)
        # emb = tf.layers.dense(emb, self.global_dimension, tf.nn.relu)
        return current_item_embedding

    def read_memory(self, user_id, item_id):
        self.memory_batch_read = tf.nn.embedding_lookup(self.memory, user_id)
        batch_key = tf.expand_dims(self.feature_key, axis=0)
        # current_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, item_id)
        current_item_embedding = self.lookup_item_embedding()
        self.w = tf.reduce_sum(tf.multiply(batch_key, tf.expand_dims(current_item_embedding, axis=1)), axis=2)
        self.weight = tf.nn.softmax(tf.expand_dims(self.w, axis=2))
        out = tf.reduce_sum(tf.multiply(self.memory_batch_read, self.weight), axis=1)
        return out

    def erase(self, i):
        out = tf.nn.sigmoid(i)
        return out

    def add(self, i):
        out = tf.nn.tanh(i)
        return out

    def write_memory(self, user_id, item_id, len):
        # current_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, item_id)
        current_item_embedding = self.lookup_item_embedding()
        self.memory_batch_write = tf.nn.embedding_lookup(self.memory, user_id)
        ones = tf.ones([len, self.memory_rows, self.global_dimension], tf.float32)
        self.rating_embedding = tf.nn.embedding_lookup(self.rating_embedding_matrix,
                                                       tf.subtract(tf.to_int32(self.rating), tf.constant(1)))
        e = tf.expand_dims(self.erase(current_item_embedding), axis=1)
        a = tf.expand_dims(self.add(current_item_embedding), axis=1)
        decay = tf.subtract(ones, tf.multiply(self.weight, e))
        increase = tf.multiply(self.weight, a)
        self.new_value = tf.add(tf.multiply(self.memory_batch_write, decay), increase)
        self.memory = tf.scatter_update(self.memory, user_id, self.new_value)

    def merge(self, u, m):
        merged = tf.add(u, tf.multiply(tf.constant(0.2), m))
        # emb = tf.concat([u, m], axis = 1)
        # merged = tf.layers.dense(emb, self.global_dimension, tf.nn.relu)
        return merged

    def build_model(self):
        print 'building model ...'

        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        # self.item_embedding = self.lookup_item_embedding()
        # tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        self.memory_out = self.read_memory(self.user_id, self.item_id)
        self.new_user_embedding = self.merge(self.user_embedding, self.memory_out)

        # self.user_bias = tf.nn.embedding_lookup(self.user_bias_vector, self.user_id)
        # self.item_bias = tf.nn.embedding_lookup(self.item_bias_vector, self.item_id)

        self.write_memory(self.user_id, self.item_id, self.len)

        # compute loss
        self.element_wise_mul = tf.multiply(self.new_user_embedding, self.item_embedding)
        # self.element_wise_mul = tf.concat([self.new_user_embedding, self.item_embedding], axis = 1)
        # self.element_wise_mul = tf.layers.dense(self.element_wise_mul, self.global_dimension, tf.nn.relu)
        self.element_wise_mul_drop = tf.nn.dropout(self.element_wise_mul, self.dropout_keep_prob)

        self.log_intention = tf.reduce_sum(self.element_wise_mul_drop, axis=1)
        # self.log_intention = tf.add(tf.add(tf.add(self.log_intention, self.user_bias),self.item_bias), self.global_bias)
        self.pred = tf.sigmoid(self.log_intention)
        self.intention_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.log_intention, name='sigmoid'))
        # self.intention_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(targets=self.label, logits=self.log_intention, name='sigmoid'))

        self.regular_loss = tf.add(0.0 * tf.nn.l2_loss(self.user_embedding),
                                   0.0 * tf.nn.l2_loss(self.item_embedding))
        self.intention_loss = tf.add(self.regular_loss, self.intention_loss)

        # l = len(self.test_candidates)
        # self.test_pol_matrix = tf.reshape(self.log_intention, shape=[-1, l])
        # self.top_value, self.top_index = tf.nn.top_k(self.test_pol_matrix, k=l, sorted=True)

    def test_AUC(self, loader):
        # auc_labels = []
        # auc_pred = []
        # cnt = 0
        # while cnt < loader.num_of_step:
        #     cnt += 1
        #     _, users, items, cates, users_back, items_back, cates_back, labels = loader.next()
        #     for i in range(len(users)):
        #         user_batch, item_batch, cate_batch = users[i], items[i], cates[i]
        #         _ = self.sess.run([self.memory],  # if buy
        #                           feed_dict={self.user_id: user_batch,
        #                                      self.item_id: item_batch,
        #                                      self.category_id: cate_batch,
        #                                      self.label: labels,
        #                                      self.len: len(user_batch)})  # self.batch_size
        #     p = self.sess.run(self.pred,
        #                       feed_dict={self.user_id: users_back,
        #                                  self.item_id: items_back,
        #                                  self.category_id: cates_back,
        #                                  self.len: loader.batch_size})
        #     auc_pred.extend(p)
        #     auc_labels.extend(labels)
        #     # auc_pred.append(p[0])
        #     # auc_labels.append(labels)
        # mean_auc = roc_auc_score(auc_labels, auc_pred)
        # print "AUC", mean_auc
        # return
        i = 0
        user_batch = []
        item_batch = []
        cate_batch = []
        user_back_batch = []
        item_back_batch = []
        labels = []
        flow_batch_size = 128
        q = Queue.Queue(maxsize=200)
        auc_labels = []
        auc_pred = []
        auc_loss = []
        cnt = 0
        # item list
        while i < 100:  # loader.num_of_step - 5:  # or not q.empty():
            ########################################################################
            # if cnt > 500:
            #     break
            ########################################################################
            if q.qsize() < flow_batch_size:
                cnt += 1
                i, (label, item_part, item_part_len, user_part, user_part_len) = loader.next()
                tt = np.array_split(user_part[0], 2, axis=1)
                q.put((item_part_len[0], 0, tt[0], tt[1], label[0]))  # user item category
                continue

            front = q.get()
            if front[1] >= front[0]:
                continue
            q.put((front[0], front[1] + 1, front[2], front[3], front[4]))
            # print front #, front[2][front[1]][0]
            # sys.exit(0)
            user_batch.append(front[2][front[1]][0])
            item_batch.append(front[3][front[1]][0])
            # cate_batch.append(front[5][front[1]][0])
            if len(user_batch) == flow_batch_size:
                _, b = self.sess.run([self.memory, self.log_intention],  # if buy
                                     feed_dict={self.user_id: user_batch,
                                                self.item_id: item_batch,
                                                # self.category_id: cate_batch,
                                                self.len: flow_batch_size})  # self.batch_size
                item_batch = []
                user_batch = []
                cate_batch = []
            if front[0] - 1 == front[1]:
                p, loss = self.sess.run([self.pred, self.intention_loss],
                                        feed_dict={self.user_id: front[2][front[1]],
                                                   self.item_id: front[3][front[1]],
                                                   # self.category_id: front[3][front[1]],
                                                   self.label: [front[4]],
                                                   self.len: 1})
                auc_pred.append(p[0])
                auc_labels.append(front[4])
                auc_loss.append(loss)
                # user_back_batch.append(front[2][front[1]][0])
                # item_back_batch.append(front[3][front[1]][0])
                # labels.append(front[4])
                # auc_labels.append(front[4])
                # if len(user_back_batch) == flow_batch_size:
                #     # print i, q.qsize()
                #     p = self.sess.run(self.pred,
                #                   feed_dict={self.user_id: user_back_batch, self.item_id: item_back_batch,
                #                              self.label: labels,
                #                              self.len: flow_batch_size})
                #     user_back_batch = []
                #     item_back_batch = []
                #     labels = []
                #     if len(auc_pred) == 0:
                #         auc_pred = p.tolist()
                #     else:
                #         auc_pred = auc_pred + p.tolist()
        mean_auc = roc_auc_score(auc_labels, auc_pred)
        print "TEST AUC", mean_auc
        return mean_auc, sum(auc_loss) / len(auc_loss)
        # self.write_log("test", sum(auc_loss) / len(auc_loss) / flow_batch_size, mean_auc)
        # if i % 500 == 0:
        #     r = self.evaluate()

        while i < loader.num_of_step - 5:  # or not q.empty():
            ########################################################################
            # if cnt > 500:
            #     break
            ########################################################################
            if q.qsize() < flow_batch_size:
                cnt += 1
                i, (label, item_part, item_part_len, user_part, user_part_len) = loader.next()
                tt = np.array_split(item_part[0], 3, axis=1)
                q.put((item_part_len[0], 0, tt[0], tt[1], label[0], tt[2]))  # user item category
                continue

            front = q.get()
            if front[1] >= front[0]:
                continue
            q.put((front[0], front[1] + 1, front[2], front[3], front[4], front[5]))
            # print front #, front[2][front[1]][0]
            # sys.exit(0)
            user_batch.append(front[2][front[1]][0])
            item_batch.append(front[3][front[1]][0])
            cate_batch.append(front[5][front[1]][0])
            if len(user_batch) == flow_batch_size:
                _, b = self.sess.run([self.memory, self.log_intention],  # if buy
                                     feed_dict={self.user_id: user_batch,
                                                self.item_id: item_batch,
                                                self.category_id: cate_batch,
                                                self.len: flow_batch_size})  # self.batch_size
                item_batch = []
                user_batch = []
                cate_batch = []
            if front[0] - 1 == front[1]:
                p, loss = self.sess.run([self.pred, self.intention_loss],
                                        feed_dict={self.user_id: front[2][front[1]],
                                                   self.item_id: front[3][front[1]],
                                                   self.category_id: front[3][front[1]],
                                                   self.label: [front[4]],
                                                   self.len: 1})
                auc_pred.append(p[0])
                auc_labels.append(front[4])
                auc_loss.append(loss)
                # user_back_batch.append(front[2][front[1]][0])
                # item_back_batch.append(front[3][front[1]][0])
                # labels.append(front[4])
                # auc_labels.append(front[4])
                # if len(user_back_batch) == flow_batch_size:
                #     # print i, q.qsize()
                #     p = self.sess.run(self.pred,
                #                   feed_dict={self.user_id: user_back_batch, self.item_id: item_back_batch,
                #                              self.label: labels,
                #                              self.len: flow_batch_size})
                #     user_back_batch = []
                #     item_back_batch = []
                #     labels = []
                #     if len(auc_pred) == 0:
                #         auc_pred = p.tolist()
                #     else:
                #         auc_pred = auc_pred + p.tolist()
        mean_auc = roc_auc_score(auc_labels, auc_pred)
        print "TEST AUC", mean_auc
        return mean_auc, sum(auc_loss) / len(auc_loss)
        # self.write_log("test", sum(auc_loss) / len(auc_loss) / flow_batch_size, mean_auc)
        # if i % 500 == 0:
        #     r = self.evaluate()

    def write_log(self, step, label, loss, auc):

        fi = open("rum_log_test.txt", "a")
        if label == "train":
            fi.write(label + " " + str(loss) + "\n")
        else:
            fi.write(label + " " + str(loss) + " " + str(auc) + "\n")
        fi.close()

    def run(self):
        print 'running ...'
        max = 0
        max_result = []
        #    def __init__(self, dataset, batch_size):
        # with open("../data/dataset.pkl", 'rb') as f:
        with open("/home/guorui.xgr/data/amazon/Electronics/dataset.pkl", 'rb') as f:
            train_set = pkl.load(f)
            test_set = pkl.load(f)
            feature_size = pkl.load(f)
            max_len = 100
            item_part_fnum = 3
            user_part_fnum = 2

        batch_size = 8
        rumloader = RUMDataloader(train_set, batch_size)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # with tf.Session(config=config) as sess:

        # with tf.Session(config=config) as self.sess:
        #     self.intention_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta2=0.99).minimize(self.intention_loss)
        #     # self.intention_optimizer = tf.train.Ada(learning_rate=0.001).minimize(
        #     #     self.intention_loss)
        #     init = tf.initialize_all_variables()
        #     self.sess.run(init)
        #     data_index = 0
        #     while data_index < rumloader.num_of_step:
        #         data_index, users, items, cates, users_back, items_back, cates_back, labels = rumloader.next()
        #         for i in range(len(users)):
        #             user_batch, item_batch, cate_batch = users[i], items[i], cates[i]
        #             # print user_batch, item_batch, cate_batch, len(user_batch)
        #             _ = self.sess.run([self.memory],  # if buy
        #                               feed_dict={self.user_id: user_batch,
        #                                          self.item_id: item_batch,
        #                                          self.category_id: cate_batch,
        #                                          self.len: len(user_batch)})  # self.batch_size
        #         self.sess.run([self.intention_optimizer], feed_dict={self.user_id: users_back,
        #                                                              self.item_id: items_back,
        #                                                              self.category_id: cates_back,
        #                                                              self.label: labels,
        #                                                              self.len: batch_size})
        #         if data_index % 10 == 0:
        #             print data_index
        #             self.test_AUC(RUMDataloader(test_set, batch_size))
        # sys.exit(0)

        loader = DataLoader(train_set, 1)
        # i, (label, item_part, item_part_len, user_part, user_part_len) = loader.next()
        # print "####"
        # print label, user_part[0] #item_part uid iid cid  user_part iid uid
        # tt = np.array_split(item_part[0], 3, axis = 1)
        # print item_part[0], user_part[0]
        # print "####"
        # sys.exit(0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:
        with tf.Session(config=config) as self.sess:
            # self.intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(
            #     self.intention_loss)

            self.intention_optimizer = tf.train.AdamOptimizer(learning_rate=5e-2, beta2=0.99).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            i = 0
            user_batch = []
            item_batch = []
            cate_batch = []
            user_back_batch = []
            item_back_batch = []
            cate_back_batch = []
            train_batch_loss = []
            labels = []
            flow_batch_size = 8
            q = Queue.Queue(maxsize=flow_batch_size + 10)
            epoch = 0
            step = 0
            print "num_of_step", loader.num_of_step
            run_time = []
            load_time = []
            last_test_step = 0
            last_clear_step = 0
            while True:  # i < loader.num_of_step:
                start_time = time.time()
                if i == loader.num_of_step - 3:
                    loader.i = 0
                    epoch += 1
                    if step != last_clear_step:
                        self.clear_memory()
                        last_clear_step = step
                    # if epoch > 9:
                    #     break
                if q.qsize() < flow_batch_size:
                    step += 1
                    if step != 0 and step % 100000 == 0:
                        print "run_time", sum(run_time), "load_time", sum(load_time)
                        run_time = []
                        load_time = []
                    i, (label, item_part, item_part_len, user_part, user_part_len) = loader.next()
                    tt = np.array_split(user_part[0], 3, axis=1)
                    q.put((user_part_len[0], 0, tt[0], tt[1], label[0]))
                    continue

                front = q.get()
                if front[1] >= front[0]:
                    continue
                q.put((front[0], front[1] + 1, front[2], front[3], front[4]))
                # print front #, front[2][front[1]][0]
                # sys.exit(0)
                user_batch.append(front[2][front[1]][0])
                item_batch.append(front[3][front[1]][0])
                # cate_batch.append(front[5][front[1]][0])
                start_run_time = time.time()
                if len(user_batch) == flow_batch_size:
                    _ = self.sess.run([self.memory],  # if buy
                                      feed_dict={self.user_id: user_batch,
                                                 self.item_id: item_batch,
                                                 # self.category_id:cate_batch,
                                                 # self.label: labels,
                                                 # self.label: [1] * flow_batch_size,
                                                 self.len: flow_batch_size})  # self.batch_size
                    item_batch = []
                    user_batch = []
                    # cate_batch = []

                if front[0] - 1 == front[1]:
                    user_back_batch.append(front[2][front[1]][0])
                    item_back_batch.append(front[3][front[1]][0])
                    # cate_back_batch.append(front[5][front[1]][0])
                    labels.append(front[4])
                    if len(user_back_batch) == flow_batch_size:
                        # print i, q.qsize  ()
                        _, loss = self.sess.run([self.intention_optimizer, self.intention_loss],
                                                feed_dict={self.user_id: user_back_batch,
                                                           self.item_id: item_back_batch,
                                                           # self.category_id: cate_back_batch,
                                                           self.label: labels,
                                                           self.len: flow_batch_size})
                        train_batch_loss.append(loss)
                        user_back_batch = []
                        item_back_batch = []
                        # cate_back_batch = []
                        labels = []
                        self.write_log(step, "train", loss / flow_batch_size, 0)
                end_run_time = time.time()
                run_time.append(end_run_time - start_run_time)
                load_time.append(start_run_time - start_time)
                if len(train_batch_loss) == flow_batch_size:
                    # self.write_log(step, "train", sum(train_batch_loss) / flow_batch_size / flow_batch_size, 0)
                    train_batch_loss = []
                if epoch > 10 and step % 10000 == 0 and last_test_step != step:
                    print epoch, step
                    last_test_step = step
                    test_loader = DataLoader(test_set, 1)
                    ttauc, ttloss = self.test_AUC(test_loader)
                    self.write_log(step, "test", ttloss, ttauc)
            return

            i = 0
            user_batch = []
            item_batch = []
            cate_batch = []
            user_back_batch = []
            item_back_batch = []
            cate_back_batch = []
            train_batch_loss = []
            labels = []
            flow_batch_size = 1280
            q = Queue.Queue(maxsize=flow_batch_size + 10)
            epoch = 0
            step = 0
            print "num_of_step", loader.num_of_step
            run_time = []
            load_time = []
            last_test_step = 0
            last_clear_step = 0
            while True:  # i < loader.num_of_step:
                start_time = time.time()
                if i == loader.num_of_step - 3:
                    loader.i = 0
                    epoch += 1
                    if step != last_clear_step:
                        self.clear_memory()
                        last_clear_step = step
                    # if epoch > 9:
                    #     break
                if q.qsize() < flow_batch_size:
                    step += 1
                    if step != 0 and step % 100000 == 0:
                        print "run_time", sum(run_time), "load_time", sum(load_time)
                        run_time = []
                        load_time = []
                    i, (label, item_part, item_part_len, user_part, user_part_len) = loader.next()
                    tt = np.array_split(item_part[0], 3, axis=1)
                    q.put((item_part_len[0], 0, tt[0], tt[1], label[0], tt[2]))
                    continue

                front = q.get()
                if front[1] >= front[0]:
                    continue
                q.put((front[0], front[1] + 1, front[2], front[3], front[4], front[5]))
                # print front #, front[2][front[1]][0]
                # sys.exit(0)
                user_batch.append(front[2][front[1]][0])
                item_batch.append(front[3][front[1]][0])
                cate_batch.append(front[5][front[1]][0])
                start_run_time = time.time()
                if len(user_batch) == flow_batch_size:
                    _ = self.sess.run([self.memory],  # if buy
                                      feed_dict={self.user_id: user_batch,
                                                 self.item_id: item_batch,
                                                 self.category_id: cate_batch,
                                                 # self.label: labels,
                                                 # self.label: [1] * flow_batch_size,
                                                 self.len: flow_batch_size})  # self.batch_size
                    item_batch = []
                    user_batch = []
                    cate_batch = []

                if front[0] - 1 == front[1]:
                    user_back_batch.append(front[2][front[1]][0])
                    item_back_batch.append(front[3][front[1]][0])
                    cate_back_batch.append(front[5][front[1]][0])
                    labels.append(front[4])
                    if len(user_back_batch) == flow_batch_size:
                        # print i, q.qsize  ()
                        _, loss = self.sess.run([self.intention_optimizer, self.intention_loss],
                                                feed_dict={self.user_id: user_back_batch,
                                                           self.item_id: item_back_batch,
                                                           self.category_id: cate_back_batch,
                                                           self.label: labels,
                                                           self.len: flow_batch_size})
                        train_batch_loss.append(loss)
                        user_back_batch = []
                        item_back_batch = []
                        cate_back_batch = []
                        labels = []
                        self.write_log(step, "train", loss / flow_batch_size, 0)
                end_run_time = time.time()
                run_time.append(end_run_time - start_run_time)
                load_time.append(start_run_time - start_time)
                if len(train_batch_loss) == flow_batch_size:
                    # self.write_log(step, "train", sum(train_batch_loss) / flow_batch_size / flow_batch_size, 0)
                    train_batch_loss = []

                if epoch > 8 and step % 100000 == 0 and last_test_step != step:
                    print epoch, step
                    last_test_step = step
                    test_loader = DataLoader(test_set, 1)
                    ttauc, ttloss = self.test_AUC(test_loader)
                    self.write_log(step, "test", ttloss, ttauc)

            # for iter in range(self.iteration):
            #     print 'new iteration begin ...'
            #     print 'iteration: ' + str(iter)
            #     print self.dg.records_number
            #
            #     while self.step * self.batch_size <= self.dg.records_number:
            #         user_batch, item_batch, rating_batch, label = self.dg.gen_train_batch_data(self.batch_size)
            #         if label[0] == 1:
            #             a, b, _ = self.sess.run([self.label, self.pred, self.intention_optimizer], feed_dict={self.user_id: user_batch, self.item_id: item_batch,
            #                                                  self.label: label,
            #                                                  self.len: self.batch_size})
            #             self.sess.run([self.memory],#if buy
            #                           feed_dict={self.user_id: user_batch, self.item_id: item_batch,
            #                                      self.rating: rating_batch, self.label: label,
            #                                      self.len: self.batch_size})
            #         else:
            #             self.sess.run(self.intention_optimizer,
            #                           feed_dict={self.user_id: user_batch, self.item_id: item_batch,
            #                                      self.rating: rating_batch, self.label: label,
            #                                      self.len: self.batch_size})
            #         self.step += 1
            #         if self.step * self.batch_size % 5000 == 0:
            #             print 'evel ...'
            #             r = self.evaluate()
            #             if r[3] > max:
            #                 max = r[3]
            #                 max_result = r
            #
            #     #self.clear_memory()
            #     self.step = 0

            self.save()
            # print max_result
            print("Optimization Finished!")

    def save(self):
        item_latent_factors = self.sess.run(self.item_embedding_matrix)
        t = pd.DataFrame(item_latent_factors)
        t.to_csv('item_latent_factors')
        feature_keys = self.sess.run(self.feature_key)
        t = pd.DataFrame(feature_keys)
        t.to_csv('feature_keys')

    def NDCG_k(self, recommend_list, purchased_list):
        user_number = len(recommend_list)
        u_ndgg = []
        for i in range(user_number):
            temp = 0
            Z_u = 0
            for j in range(len(recommend_list[i])):
                Z_u = Z_u + 1 / np.log2(j + 2)
                if recommend_list[i][j] in purchased_list[i]:
                    temp = temp + 1 / np.log2(j + 2)
            if Z_u == 0:
                temp = 0
            else:
                temp = temp / Z_u
            u_ndgg.append(temp)
        return u_ndgg

    def top_k(self, pre_top_k, true_top_k):
        user_number = len(pre_top_k)
        correct = []
        co_length = []
        re_length = []
        pu_length = []
        p = []
        r = []
        f = []
        hit = []
        for i in range(user_number):
            temp = []
            for j in pre_top_k[i]:
                if j in true_top_k[i]:
                    temp.append(j)
            if len(temp):
                hit.append(1)
            else:
                hit.append(0)
            co_length.append(len(temp))
            re_length.append(len(pre_top_k[i]))
            pu_length.append(len(true_top_k[i]))
            correct.append(temp)

        # print co_length

        for i in range(user_number):
            if re_length[i] == 0:
                p_t = 0.0
            else:
                p_t = co_length[i] / float(re_length[i])
            if pu_length[i] == 0:
                r_t = 0.0
            else:
                r_t = co_length[i] / float(pu_length[i])
            p.append(p_t)
            r.append(r_t)
            if p_t != 0 or r_t != 0:
                f.append(2.0 * p_t * r_t / (p_t + r_t))
            else:
                f.append(0.0)
        return p, r, f, hit

    def evaluate(self):
        user_number = 1
        all_p = []
        all_r = []
        all_f1 = []
        all_hit_ratio = []
        all_ndcg = []

        for i in range(len(self.test_users) / user_number):
            batch_users, batch_items = self.dg.gen_test_batch_data(user_number)
            top_k_value, top_k_index = self.sess.run(
                [self.top_value, self.top_index],
                feed_dict={self.user_id: batch_users,
                           self.item_id: batch_items})

            pre_top_k = []
            ground_truth = []

            user_index_begin = i * user_number
            user_index_end = (i + 1) * user_number

            for user_index in range(user_index_begin, user_index_end):
                index = [j for j in top_k_index[user_index - user_index_begin] if
                         self.test_candidates[j] not in self.train_user_purchsed_items_dict[
                             self.test_users[user_index]]]
                items = [self.test_candidates[j] for j in index]

                pre_top_k.append(list(items[:self.K]))
                ground_truth.append([k for k in self.test_user_purchsed_items_dict[self.test_users[user_index]] if
                                     k in self.test_candidates])

            p, r, f1, hit_ratio = self.top_k(pre_top_k, ground_truth)
            ndcg = self.NDCG_k(pre_top_k, ground_truth)

            all_p.append(np.array(p).mean())
            all_r.append(np.array(r).mean())
            all_f1.append(np.array(f1).mean())
            all_hit_ratio.append(np.array(hit_ratio).mean())
            all_ndcg.append(np.array(ndcg).mean())

        self.logger.info("Presicion@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_p).mean()))
        self.logger.info("Recall@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_r).mean()))
        self.logger.info("F1@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_f1).mean()))
        self.logger.info("hit@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_hit_ratio).mean()))
        self.logger.info("NDCG@" + str(self.K) + "= " + "{:.6f}".format(np.array(all_ndcg).mean()))

        return [np.array(all_p).mean(), np.array(all_r).mean(), np.array(all_f1).mean(), np.array(all_hit_ratio).mean(),
                np.array(all_ndcg).mean()]


if __name__ == '__main__':
    type = 'rating_instant_video_raw10_0.7'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    m = rum(type)
    m.build_model()
    m.run()
