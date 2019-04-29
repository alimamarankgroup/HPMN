from srnn import *
import cPickle as pkl
import sys
import os
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("PLEASE INPUT [DATASET] and [GPU]")
        sys.exit(0)
    dataset = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

    if dataset == 'taobao':
        with open('../data/taobao/dataset_crop.pkl', 'rb') as f:
            train_set = pkl.load(f)
            test_set = pkl.load(f)
            feature_size = pkl.load(f)
            user_mem_maxlen = 256
            item_mem_maxlen = 27
            user_que_maxlen = 44
            item_que_maxlen = 8   

            item_part_fnum = 3
            user_part_fnum = 4
            user, item, dual = True, False, False
            model = shan('model/shan_taobao', train_set, test_set, 2, user_mem_maxlen, item_mem_maxlen, user_que_maxlen, item_que_maxlen,
                    feature_size, user, item, dual, user_feature_number=4, item_feature_number=3, learning_rate=1e-3, hidden_size=36,
                    embedding_size=18, batchsize=10, beta=0, hop=1, startpoint=0)
            model.train(6)
    elif dataset == 'amazon':
          with open('../data/amazon/dataset_crop.pkl', 'rb') as f:
              train_set = pkl.load(f)
              test_set = pkl.load(f)
              feature_size = pkl.load(f)
              user_mem_maxlen = 90
              item_mem_maxlen = 90
              user_que_maxlen = 10
              item_que_maxlen = 10   

              item_part_fnum = 2
              user_part_fnum = 3
              user, item, dual = True, False, False
              model = shan('model/shan_amazon', train_set, test_set, 2, user_mem_maxlen, item_mem_maxlen, user_que_maxlen, item_que_maxlen,
                       feature_size, user, item, dual, user_feature_number=3, item_feature_number=2, learning_rate=1e-3, hidden_size=36,
                        embedding_size=18, batchsize=10, beta=0, hop=1, startpoint=0)
              model.train(6)

