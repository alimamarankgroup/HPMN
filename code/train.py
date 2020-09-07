import os
import cPickle as pkl
import tensorflow as tf
import sys
from data_loader import *
from model import *
from sklearn.metrics import *
import random
import numpy as np
import time
from util import *
import numpy as np

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2

EVAL_BATCH_SIZE = 1024

def eval(model, sess, dataset, islong):
    preds = []
    labels = []
    users = []
    if not isinstance(dataset, str):
        if islong:
            dataloader = DataLoader(dataset, EVAL_BATCH_SIZE)
        else:
            dataloader = CroppedLoader(dataset, EVAL_BATCH_SIZE)
    else:
        if islong:
            dataloader = DataLoader_Mul(dataset, EVAL_BATCH_SIZE)
        else:
            dataloader = DataLoader_crop(dataset, EVAL_BATCH_SIZE)

    t = time.time()
    for _, batch_data in dataloader:
        if not islong:
            batch_data = (batch_data[0], batch_data[2], batch_data[4], batch_data[6], batch_data[8])
        pred, label = model.eval(sess, batch_data)
        users += batch_data[1][:, 0, 0].tolist()
        preds += pred
        labels += label
    print("EVAL TIME: %.4fs" % (time.time() - t))
    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    # gauc = calculate_gauc(labels, preds, users)
    return logloss, auc

def train(model_type, train_num, train_set, test_set, feature_size, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, islong, train_batch_size, lr, reg_lambda, iter_num, dataset_name, emb_initializer=None):
    if model_type == 'DNN':
        model = DNN(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    elif model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    elif model_type == 'ARNN':
        model = ARNN(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    elif model_type == 'DIEN':
        model = DIEN(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    elif model_type == 'CASER':
        model = CASER(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    elif model_type == 'RUM_ITEM':
        model = RUM_ITEM(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    elif model_type == 'SVD++':
        model = SVDpp(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    elif model_type == 'LSTM':
        model = LSTM4Rec(feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_len_item, max_len_user, item_part_fnum, user_part_fnum, use_hist_u, use_hist_i, emb_initializer)
    else:
        print('WRONG MODEL TYPE')
        sys.exit(1)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses = []
        test_losses = []

        test_aucs = []
        train_aucs = []

        # before training process
        step = 0
        train_loss, train_auc = eval(model, sess, train_set, islong)
        test_loss, test_auc = eval(model, sess, test_set, islong)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

        print("STEP %d  LOSS_TRAIN: %.4f  LOSS_TEST: %.4f  AUC_TRAIN: %.4f  AUC_TEST: %.4f" % (step, train_loss, test_loss, train_auc, test_auc))
        early_stop = False

        # begin training process
        for epoch in range(4):
            if early_stop:
                break
            if not isinstance(train_set, str):
                random.shuffle(train_set)
                if islong:
                    dataloader = DataLoader(train_set, train_batch_size)
                else:
                    dataloader = CroppedLoader(train_set, train_batch_size)
            else:
                if islong:
                    dataloader = DataLoader_Mul(train_set, train_batch_size)
                else:
                    dataloader = DataLoader_crop(train_set, train_batch_size)
            
            
            for _, batch_data in dataloader:
                if early_stop:
                    break

                if not islong:
                    batch_data = (batch_data[0], batch_data[2], batch_data[4], batch_data[6], batch_data[8])

                loss = model.train(sess, batch_data, lr, reg_lambda)
                # train_losses_step.append(loss)
                step += 1

                if step % iter_num == 0:
                    train_loss, train_auc = eval(model, sess, train_set, islong)
                    test_loss, test_auc = eval(model, sess, test_set, islong)

                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    train_aucs.append(train_auc)
                    test_aucs.append(test_auc)

                    print("STEP %d  LOSS_TRAIN: %.4f  LOSS_TEST: %.4f  AUC_TRAIN: %.4f  AUC_TEST: %.4f" % (step, train_loss, test_loss, train_auc, test_auc))
                if len(test_losses) > 2 and epoch > 0:
                    if (test_losses[-1] > test_losses[-2] and test_losses[-2] > test_losses[-3]):
                        early_stop = True

        # save model
        # model_name = '{}_{}_{}_{}_{}_{}_{}'.format(model_type, use_hist_u, use_hist_i, islong, train_batch_size, lr, reg_lambda)
        # if not os.path.exists('save_model_ind/{}/'.format(model_name)):
        #     os.mkdir('save_model_ind/{}/'.format(model_name))
        # save_path = 'save_model_ind/{}/ckpt'.format(model_name)
        # model.save(sess, save_path)
        
        logname = '{}_{}_{}_{}.log'.format(model_type, train_batch_size, lr, reg_lambda)
        if not os.path.exists('logs_{}'.format(dataset_name)):
            os.mkdir('logs_{}'.format(dataset_name))
        with open('logs_{}/{}'.format(dataset_name, logname), 'wb') as f:
            dump_tuple = (train_losses, test_losses, train_aucs, test_aucs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(dataset_name, logname), 'w') as f:
            f.write('Best Test AUC: {}\n'.format(max(test_aucs)))
            f.write('Best Test Logloss: {}\n'.format(test_losses[np.argmax(test_aucs)]))
        
        return max(test_aucs), test_losses[np.argmax(test_aucs)]


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("PLEASE INPUT [MODEL TYPE], [DATASET] and [GPU]")
        sys.exit(0)
    model_type = sys.argv[1]
    dataset = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
    
    use_hist_u = False
    use_hist_i = True
    islong = True

    if dataset == 'amazon':
        with open('../data/amazon/dataset.pkl', 'rb') as f:
            train_set = pkl.load(f)
            test_set = pkl.load(f)
            feature_size = pkl.load(f)+1
            max_len_item = 100
            max_len_user = 100
            item_part_fnum = 3
            user_part_fnum = 2
            emb_initializer = None
            iter_num = 2 #change this when you do the expriments on the full datasets
    elif dataset == 'taobao':
        with open('../data/taobao/dataset.pkl', 'rb') as f:
            train_set = pkl.load(f)
            test_set = pkl.load(f)
            feature_size = pkl.load(f)+1
            max_len_item = 300
            max_len_user = 35
            item_part_fnum = 4
            user_part_fnum = 3
            emb_initializer = None
            iter_num = 1
    elif dataset == 'xlong':
        train_set = '../data/xlong/train_corpus_total_dual.txt'
        test_set = '../data/xlong/test_corpus_total_dual.txt'
        pv_cnt = 19002
        feature_size = pv_cnt + np.load('../data/xlong/graph_emb.npy').shape[0] + 20000
        max_len_item = 1000 + 1
        max_len_user = 184
        item_part_fnum = 2
        user_part_fnum = 1
        
        item_emb_init = np.load('../data/xlong/graph_emb.npy')
        user_emb_init = np.zeros([20000, 16])
        pv_num_emb = np.zeros([pv_cnt, 16])
        emb_initializer = np.concatenate((item_emb_init, user_emb_init, pv_num_emb), 0).astype(np.float32)
        iter_num = 80

    # hyperparameters: train batch size, learning rate
    train_batch_sizes = [128]
    lrs = [1e-3, 5e-3, 1e-4]
    reg_lambdas = [1e-3, 1e-4, 1e-5]
    
    result_auc = 0
    result_logloss = 10
    repeat_times = 1
    for train_batch_size in train_batch_sizes:
        for lr in lrs:
            for reg_lambda in reg_lambdas:
                test_aucs = []
                test_loglosses = []
                for i in range(repeat_times):
                    test_auc, test_logloss = train(model_type, i, train_set, test_set, feature_size, max_len_item, max_len_user, item_part_fnum, 
                                                    user_part_fnum, use_hist_u, use_hist_i, islong, train_batch_size, lr, reg_lambda, iter_num, dataset, emb_initializer)
                    test_aucs.append(test_auc)
                    test_loglosses.append(test_logloss)

                if sum(test_aucs) / repeat_times > result_auc:
                    result_auc = sum(test_aucs) / repeat_times
                if sum(test_loglosses) / repeat_times < result_logloss:
                    result_logloss = sum(test_loglosses) / repeat_times
    print("FINAL RESULT: AUC=%.4f\tLOGLOSS=%.4f" % (result_auc, result_logloss))
