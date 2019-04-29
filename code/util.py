import os

from sklearn.metrics import *
import sys
if sys.version_info < (3, 0):
    import cPickle as pkl
else:
    import pickle as pkl
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
import gc
import numpy as np
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn_cell_impl import _Linear

# DATASET_PKL_PATH = '../../data/taobao/dataset.pkl'
# CROP_PKL_PATH = '../../data/taobao/dataset_crop.pkl'
# PADDING_PKL_PATH = '../../data/taobao/dataset_fp.pkl'

DATASET_PKL_PATH = '../../data/amazon/Electronics/dataset.pkl'
CROP_PKL_PATH = '../../data/amazon/Electronics/dataset_crop.pkl'
PADDING_PKL_PATH = '../../data/amazon/Electronics/dataset_fp.pkl'


def calculate_gauc(labels, preds, users):
    user_aucs = []
    user_pred_dict = {}
    for i in range(len(users)):
        if users[i] in user_pred_dict:
            user_pred_dict[users[i]][0].append(preds[i])
            user_pred_dict[users[i]][1].append(labels[i])
        else:
            user_pred_dict[users[i]] = ([preds[i]], [labels[i]])
    # calculate
    for u in user_pred_dict:
        user_aucs.append(len(user_pred_dict[u][1]) * roc_auc_score(user_pred_dict[u][1], user_pred_dict[u][0]))
    return sum(user_aucs) / len(users)


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def crop(seq, user_short, user_total, user_dim, item_short, item_total, item_dim):
    [
        label, user_seq, user_len, item_seq, item_len
    ] = seq

    if user_len < user_short:
        user_que = user_seq[:user_len] + [[0] * user_dim for i in range(user_short - user_len)]
        user_mem = [[0] * user_dim for i in range(user_total - user_short)]
        user_mem_len = 0
        user_que_len = user_len
    else:
        user_mem = [[0] * user_dim for i in range(user_total - user_len)] + user_seq[:user_len - user_short]
        user_que = user_seq[user_len - user_short: user_len]
        user_mem_len = user_len - user_short
        user_que_len = user_short
    if item_len < item_short:
        item_que = item_seq[:item_len] + [[0] * item_dim for i in range(item_short - item_len)]
        item_mem = [[0] * item_dim for i in range(item_total - item_short)]
        item_mem_len = 0
        item_que_len = item_len
    else:
        item_mem = [[0] * item_dim for i in range(item_total - item_len)] + item_seq[:item_len - item_short]
        item_que = item_seq[item_len - item_short: item_len]
        item_mem_len = item_len - item_short
        item_que_len = item_short

    return label, user_mem, user_que, user_mem_len, user_que_len, item_mem, item_que, item_mem_len, item_que_len

def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in np.arange(N)], axis=dim)

def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

def front_padding(seq, user_max, user_dim, item_max, item_dim):
    [
        label, user_seq, user_len, item_seq, item_len
    ] = seq
    user_seq = [[0] * user_dim for i in range(user_max - user_len)] + user_seq[:user_len]
    item_seq = [[0] * item_dim for i in range(item_max - item_len)] + item_seq[:item_len]

    return label, user_seq, user_len, item_seq, item_len


def get_best_result(path):
    fins = []
    for root, _, files in os.walk(path):
        for file in files:
            if file == root[len(path):].split('_')[0]:
                fins.append(os.path.join(root, file))
    best_auc = 0
    best_result = ''
    for fin in fins:
        best = 0
        bs = 0
        f = open(fin, 'r')
        for line in f.readlines():
            try:
                dataset, step, loss, auc = line.split('\t')
            except:
                continue
            if dataset == 'test' and float(auc) > best_auc:
                best_auc = float(auc)
                best_result = fin
            if dataset == 'test' and float(auc) > best:
                best = float(auc)
                bs = step
        # if float(best) > 0.86:
        print(best, fin, bs)
    print(best_auc, best_result)

