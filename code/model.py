import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from util import VecAttGRUCell
from rnn import dynamic_rnn
import numpy as np

class Model(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, 
                 use_hist_u, use_hist_i, emb_initializer):
        # reset graph
        tf.reset_default_graph()
        self.emb_initializer = emb_initializer
        
        # Input placeholders
        with tf.name_scope('Inputs'):
            self.item_part_ph = tf.placeholder(tf.int32, [None, max_len_item, item_part_fnum], name='item_part_ph')
            self.item_len_ph = tf.placeholder(tf.int32, [None,], name='item_len_ph')
            self.user_part_ph = tf.placeholder(tf.int32, [None, max_len_user, user_part_fnum], name='user_part_ph')
            self.user_len_ph = tf.placeholder(tf.int32, [None,], name='user_len_ph')
            self.label_ph = tf.placeholder(tf.float32, [None,], name='label_ph')
            self.keep_prob = tf.placeholder(tf.float32)

            # learning rate
            self.lr = tf.placeholder(tf.float64, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])

        # Embedding Layer
        with tf.name_scope('Embedding'):
            if self.emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=self.emb_initializer)
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim])
            self.item_part_emb = tf.nn.embedding_lookup(self.emb_mtx, self.item_part_ph) # [B, T, F, EMB]
            self.user_part_emb = tf.nn.embedding_lookup(self.emb_mtx, self.user_part_ph)
            # reshape
            item_oshape = self.item_part_emb.get_shape().as_list()
            user_oshape = self.user_part_emb.get_shape().as_list()

            self.item_part_emb = tf.reshape(self.item_part_emb, [-1, item_oshape[1], item_oshape[2] * user_oshape[3]]) # [B, T, F*EMB]
            self.user_part_emb = tf.reshape(self.user_part_emb, [-1, user_oshape[1], user_oshape[2] * user_oshape[3]]) # [B, T, F*EMB]


        # Item and User Mask
        self.item_mask = tf.sequence_mask(self.item_len_ph, tf.shape(self.item_part_emb)[1], dtype=tf.float32) # [B, T]
        self.item_mask = tf.expand_dims(self.item_mask, -1) # [B, T, 1]
        self.item_mask_1 = tf.sequence_mask(self.item_len_ph - 1, tf.shape(self.item_part_emb)[1], dtype=tf.float32) # [B, T]
        self.item_mask_1 = tf.expand_dims(self.item_mask_1, -1)

        self.user_mask = tf.sequence_mask(self.user_len_ph, tf.shape(self.user_part_emb)[1], dtype=tf.float32)
        self.user_mask = tf.expand_dims(self.user_mask, -1)
        self.user_mask_1 = tf.sequence_mask(self.user_len_ph - 1, tf.shape(self.user_part_emb)[1], dtype=tf.float32)
        self.user_mask_1 = tf.expand_dims(self.user_mask_1, -1)


        self.target_item = tf.reduce_sum((self.item_mask - self.item_mask_1) * self.item_part_emb, axis=1) #[B, F*EMB]
        self.target_user = tf.reduce_sum((self.user_mask - self.user_mask_1) * self.user_part_emb, axis=1)


    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.tanh, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        # output
        self.y_pred = tf.reshape(tf.nn.sigmoid(fc3), [-1,])

    def build_loss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)

        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict = {
                self.label_ph : batch_data[0],
                self.item_part_ph : batch_data[1],
                self.item_len_ph : batch_data[2],
                self.user_part_ph : batch_data[3],
                self.user_len_ph : batch_data[4],
                self.lr : lr,
                self.reg_lambda : reg_lambda,
                self.keep_prob : 0.8
            })
        return loss
    
    def eval(self, sess, batch_data):
        pred, label = sess.run([self.y_pred, self.label_ph], feed_dict = {
                self.label_ph : batch_data[0],
                self.item_part_ph : batch_data[1],
                self.item_len_ph : batch_data[2],
                self.user_part_ph : batch_data[3],
                self.user_len_ph : batch_data[4],
                self.keep_prob : 1.0
            })
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist()
    
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))
    
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
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1]) # [B, T, Dk]
        inp = tf.concat([queries, key, queries - key, queries * key], axis = -1)
        fc1 = tf.layers.dense(inp, 80, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 40, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 1, activation=None) #[B, T, 1]

        mask = tf.equal(mask, tf.ones_like(mask)) #[B, T, 1]
        paddings = tf.ones_like(fc3) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.reshape(tf.where(mask, fc3, paddings), [-1, max_len])) #[B, T]
        
        atten_output = tf.multiply(value, tf.expand_dims(score, 2))
        atten_output_sum = tf.reduce_sum(atten_output, axis=1)

        return atten_output_sum, atten_output, score
    
    def attention_v2(self, key, value, query, mask):
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None) #[B, Dk]
        fc1 = tf.layers.dense(key, k_dim, activation=tf.nn.relu) #[B, T, Dk]
        queries = tf.expand_dims(query ,1) #[B, 1, Dk]
        product = tf.reduce_sum((queries * key), axis=-1) #[B, T]

        mask = tf.reshape(tf.equal(mask, tf.ones_like(mask)), [-1, max_len]) #[B, T]
        paddings = tf.ones_like(product) * (-2 ** 32 + 1)
        score = tf.nn.softmax(tf.where(mask, product, paddings)) #[B, T]

        atten_output = tf.multiply(value, tf.expand_dims(score, 2)) #[B, T, Dv==Dk]
        atten_output_sum = tf.reduce_sum(atten_output, axis=1) #[B, Dv==Dk]

        return atten_output_sum, atten_output, score


class DNN(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(DNN, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)

        # sum pooling
        # item_part = tf.reduce_sum(self.item_part_emb * self.item_mask, 1)
        # user_part = tf.reduce_sum(self.user_part_emb * self.user_mask, 1)
        item_part = tf.concat([self.target_user, self.target_item, tf.reduce_sum(self.item_part_emb * self.item_mask_1, 1)], axis=1)
        user_part = tf.concat([self.target_user, self.target_item, tf.reduce_sum(self.user_part_emb * self.user_mask_1, 1)], axis=1) 

        if use_hist_i and use_hist_u:
            inp = tf.concat([item_part, user_part], axis=1)
        elif use_hist_i and not use_hist_u:
            inp = item_part
        elif not use_hist_i and use_hist_u:
            inp = user_part

        # fully connected layer
        self.build_fc_net(inp)
        self.build_loss()


class GRU4Rec(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(GRU4Rec, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)

        # RNN layer
        with tf.name_scope('item_rnn'):
            _, item_part_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_part_emb,
                                        sequence_length=self.item_len_ph, dtype=tf.float32, scope='gru1')
        item_part = item_part_final_state
        
        with tf.name_scope('user_rnn'):
            _, user_part_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_part_emb,
                                        sequence_length=self.user_len_ph, dtype=tf.float32, scope='gru2')
        user_part = user_part_final_state
        
        if use_hist_i and use_hist_u:
            inp = tf.concat([item_part, user_part], axis=1)
        elif use_hist_i and not use_hist_u:
            inp = item_part
        elif not use_hist_i and use_hist_u:
            inp = user_part
        
        # fully connected layer
        self.build_fc_net(inp)
        self.build_loss()

class LSTM4Rec(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(LSTM4Rec, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)

        # RNN layer
        with tf.name_scope('item_rnn'):
            _, item_part_final_state = tf.nn.dynamic_rnn(LSTMCell(hidden_size, state_is_tuple=False), inputs=self.item_part_emb,
                                        sequence_length=self.item_len_ph, dtype=tf.float32, scope='lstm1')
        item_part = item_part_final_state
        
        with tf.name_scope('user_rnn'):
            _, user_part_final_state = tf.nn.dynamic_rnn(LSTMCell(hidden_size, state_is_tuple=False), inputs=self.user_part_emb,
                                        sequence_length=self.user_len_ph, dtype=tf.float32, scope='lstm2')
        user_part = user_part_final_state
        
        if use_hist_i and use_hist_u:
            inp = tf.concat([item_part, user_part], axis=1)
        elif use_hist_i and not use_hist_u:
            inp = item_part
        elif not use_hist_i and use_hist_u:
            inp = user_part
        # fully connected layer
        self.build_fc_net(inp)
        self.build_loss()


class ARNN(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(ARNN, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)

        # attention RNN layer: Item
        with tf.name_scope('item_rnn_1'):
            item_part_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_part_emb,
                                        sequence_length=self.item_len_ph, dtype=tf.float32, scope='gru1')
        with tf.name_scope('item_attention'):
            item_part, _, _ = self.attention(item_part_output, item_part_output, self.target_item, self.item_mask)

        # attention RNN layer: User
        with tf.name_scope('user_rnn_1'):
            user_part_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_part_emb,
                                        sequence_length=self.user_len_ph, dtype=tf.float32, scope='gru2')
        with tf.name_scope('user_attention'):
            user_part, _, _ = self.attention(user_part_output, user_part_output, self.target_user, self.user_mask)

        if use_hist_i and use_hist_u:
            inp = tf.concat([item_part, user_part], axis=1)
        elif use_hist_i and not use_hist_u:
            inp = item_part
        elif not use_hist_i and use_hist_u:
            inp = user_part

        # fully connected layer
        self.build_fc_net(inp)
        self.build_loss()


class DIEN(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(DIEN, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)

        # attention RNN layer: Item
        with tf.name_scope('item_rnn_1'):
            item_part_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_part_emb,
                                        sequence_length=self.item_len_ph, dtype=tf.float32, scope='gru1')
        with tf.name_scope('item_attention'):
            _, _, item_score = self.attention(item_part_output, item_part_output, self.target_item, self.item_mask)
        with tf.name_scope('item_rnn_2'):
            _, item_part = dynamic_rnn(VecAttGRUCell(hidden_size), inputs=item_part_output,
                                                     att_scores = tf.expand_dims(item_score, -1),
                                                     sequence_length=self.item_len_ph, dtype=tf.float32,  scope="argru1")
        
        # attention RNN layer: User
        with tf.name_scope('user_rnn_1'):
            user_part_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.user_part_emb,
                                        sequence_length=self.user_len_ph, dtype=tf.float32, scope='gru2')
        with tf.name_scope('user_attention'):
            _, _, user_score = self.attention(user_part_output, user_part_output, self.target_user, self.user_mask)
        with tf.name_scope('user_rnn_2'):
            _, user_part = dynamic_rnn(VecAttGRUCell(hidden_size), inputs=user_part_output,
                                                     att_scores = tf.expand_dims(user_score, -1),
                                                     sequence_length=self.user_len_ph, dtype=tf.float32,  scope="argru2")
        if use_hist_i and use_hist_u:
            inp = tf.concat([item_part, user_part], axis=1)
        elif use_hist_i and not use_hist_u:
            inp = item_part
        elif not use_hist_i and use_hist_u:
            inp = user_part

        # fully connected layer
        self.build_fc_net(inp)
        self.build_loss()


class CASER(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(CASER, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)

        # item part hist
        with tf.name_scope('item_cnn'):
            # horizontal filters
            filters_item = 32
            h_kernel_size_item = [8, item_part_fnum * eb_dim]
            v_kernel_size_item = [self.item_part_emb.get_shape().as_list()[1], 1]

            self.item_part_emb = tf.expand_dims(self.item_part_emb, 3)
            conv1 = tf.layers.conv2d(self.item_part_emb, filters_item, h_kernel_size_item)
            max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
            item_hori_out = tf.reshape(max1, [-1, filters_item]) #[B, F]

            # vertical
            conv2 = tf.layers.conv2d(self.item_part_emb, filters_item, v_kernel_size_item)
            conv2 = tf.reshape(conv2, [-1, item_part_fnum * eb_dim, filters_item])
            item_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, item_part_fnum * eb_dim])

            item_part = tf.concat([item_hori_out, item_vert_out, self.target_user], axis=1)

        # user part hist
        with tf.name_scope('user_cnn'):
            # horizontal filters
            filters_user = 32
            h_kernel_size_user = [8, user_part_fnum * eb_dim]
            v_kernel_size_user = [self.user_part_emb.get_shape().as_list()[1], 1]

            self.user_part_emb = tf.expand_dims(self.user_part_emb, 3)
            conv1 = tf.layers.conv2d(self.user_part_emb, filters_user, h_kernel_size_user)
            max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
            user_hori_out = tf.reshape(max1, [-1, filters_user]) #[B, F]

            # vertical
            conv2 = tf.layers.conv2d(self.user_part_emb, filters_user, v_kernel_size_user)
            conv2 = tf.reshape(conv2, [-1, user_part_fnum * eb_dim, filters_user])
            user_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, user_part_fnum * eb_dim])

            user_part = tf.concat([user_hori_out, user_vert_out, self.target_item], axis=1)



        if use_hist_i and use_hist_u:
            inp = tf.concat([item_part, user_part], axis=1)
        elif use_hist_i and not use_hist_u:
            inp = item_part
        elif not use_hist_i and use_hist_u:
            inp = user_part

        # fully connected layer
        self.build_fc_net(inp)
        self.build_loss()


class RUM_ITEM(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(RUM_ITEM, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)

        # item part hist
        with tf.name_scope('item_atten'):
            item_att_sum, _, _ = self.attention(self.item_part_emb * self.item_mask_1, self.item_part_emb * self.item_mask_1, self.target_item, self.item_mask)

        # user part hist
        with tf.name_scope('user_atten'):
            user_att_sum, _, _ = self.attention(self.user_part_emb * self.user_mask_1, self.user_part_emb * self.user_mask_1, self.target_user, self.user_mask)

        if use_hist_i and use_hist_u:
            inp = tf.concat([self.target_item, self.target_user, item_att_sum, user_att_sum], axis=1)
        elif use_hist_i and not use_hist_u:
            inp = tf.concat([self.target_item, self.target_user, item_att_sum], axis=1)
        elif not use_hist_i and use_hist_u:
            inp = tf.concat([self.target_item, self.target_user, user_att_sum], axis=1)

        # fully connected layer
        self.build_fc_net(inp)
        self.build_loss()


class SVDpp(Model):
    def __init__(self, feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer):
        super(SVDpp, self).__init__(feature_size, eb_dim, hidden_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
        # projected emb seq

        self.item_seq = self.item_part_emb[:, :, eb_dim:2*eb_dim]
        self.target_item = tf.reduce_sum(self.item_seq * (self.item_mask - self.item_mask_1), axis=1)
        self.user_seq = self.item_part_emb[:, :, :eb_dim]
        self.target_user = tf.reduce_sum(self.user_seq * (self.item_mask - self.item_mask_1), axis=1)
        
        # mask
        self.item_seq_masked = self.item_seq * self.item_mask
        # svd pred
        self.neighbor = tf.reduce_sum(self.item_seq_masked, axis=1)
    
        self.norm_neighbor = self.neighbor / tf.sqrt(tf.expand_dims(tf.norm(self.item_seq_masked, 1, (1, 2)), 1))
        self.latent_score = tf.reduce_sum(self.target_item *(self.target_user + self.norm_neighbor), 1) #[B,]

        self.y_pred = tf.nn.sigmoid(self.latent_score)
        
        self.build_loss()
