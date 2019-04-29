import sys
if sys.version_info < (3, 0):
    import cPickle as pkl
else:
    import pickle as pkl
import pandas as pd
#import matplotlib.pyplot as plt
import random
from datetime import datetime
from util import crop, front_padding

RAW_DATA_FILE = '../data/raw_data/taobao/taobao_sample.csv'
SAVE_PKL_PATH = '../data/raw_data/taobao/save.pkl'
STAT_PKL_PATH = '../data/raw_data/taobao/stat.pkl'
FIG_PATH = '../data/raw_data/taobao/stat.png'

DATASET_PKL = '../data/taobao/dataset.pkl'

MAX_LEN_ITEM = 300
MAX_LEN_USER = 35

def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df

def remap(df):
    item_key = sorted(df['iid'].unique().tolist())
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))
    df['iid'] = df['iid'].map(lambda x: item_map[x])

    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(item_len, item_len + user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(user_len + item_len, user_len + item_len + cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(user_len + item_len + cate_len, user_len + item_len + cate_len + btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])

    print(item_len, user_len, cate_len, btag_len)
    return df, item_len, user_len + item_len + cate_len + btag_len + 1 #+1 is for unknown target btag


def gen_user_item_group(df, item_cnt, feature_size):
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')

    # with open(SAVE_PKL_PATH, 'wb') as f:
    #     pkl.dump(user_df, f)
    #     pkl.dump(item_df, f)
    #     pkl.dump(item_cnt, f)
    #     pkl.dump(feature_size, f)
    print("group completed")
    return user_df, item_df


def get_stat(user_df, item_df, stat_pkl):
    # with open(save_pkl, 'rb') as f:
    #     user_df = pkl.load(f)
    #     item_df = pkl.load(f)
    
    # plot statistic
    user_hist_stat = []
    item_hist_stat = []

    for uid, hist in user_df:
        length = len(hist['iid'].tolist())
        user_hist_stat.append(length)
    for iid, hist in item_df:
        length = len(hist['uid'].tolist())
        item_hist_stat.append(length)
    
    with open(stat_pkl, 'wb') as f:
        pkl.dump(user_hist_stat, f)
        pkl.dump(item_hist_stat, f)
    print('get stat completed')


def plt_stat(stat_pkl, fig_path):
    with open(stat_pkl, 'rb') as f:
        user_hist_stat = pkl.load(f)
        item_hist_stat = pkl.load(f)
    
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    edges_user = range(1, 600, 1)
    plt.hist(user_hist_stat, bins = edges_user)
    plt.xlabel('history length')
    plt.ylabel('user cnt')
    plt.title('user side history statistics')

    plt.subplot(1, 2, 2)
    edges_item = range(1, 200, 1)
    plt.hist(item_hist_stat, bins = edges_item)
    plt.xlabel('history length')
    plt.ylabel('item cnt')
    plt.title('item side history statistics')

    plt.savefig(fig_path)

def gen_dataset(user_df, item_df, item_cnt, feature_size, dataset_pkl):
    train_set = []
    test_set = []

    # get each user's last touch point time
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    for uid, hist in user_df:
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()
        target_item_time = hist['time'].tolist()[-1]

        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        target_item_btag = feature_size
        label = 1
        test = (target_item_time > split_time)

        # neg sampling
        neg = random.randint(0, 1)
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]:
                target_item = random.randint(0, item_cnt - 1)
                target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]
                target_item_btag = feature_size


        # the item history part of the sample
        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        item_part.append([uid, target_item, target_item_cate, target_item_btag])
        item_part_len = min(len(item_part), MAX_LEN_ITEM)

        # choose the item side information: which user has clicked the target item
        item_side = item_df.get_group(target_item)
        user_hist = item_side['uid'].tolist()
        user_hist_btag = item_side['btag'].tolist()
        user_hist_time = item_side['time'].tolist()

        user_part = []
        for i in range(len(user_hist)):
            if user_hist_time[i] < target_item_time:
                user_part.append([target_item, user_hist[i], user_hist_btag[i]])
        user_part.append([target_item, uid, target_item_btag])
        user_part_len = min(len(user_part), MAX_LEN_USER)

        if len(user_part) == 0:
            continue

        # padding history with 0
        if len(item_part) <= MAX_LEN_ITEM:
            item_part_pad = item_part + [[0] * 4] * (MAX_LEN_ITEM - len(item_part))
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]
        if len(user_part) <= MAX_LEN_USER:
            user_part_pad = user_part + [[0] * 3] * (MAX_LEN_USER - len(user_part))
        else:
            user_part_pad = user_part[len(user_part) - MAX_LEN_USER:len(user_part)]
        
        # gen sample
        sample = (label, item_part_pad, item_part_len, user_part_pad, user_part_len)
        
        if test:
            test_set.append(sample)
        else:
            train_set.append(sample)
    print(len(train_set))
    print(len(test_set))
    with open(dataset_pkl, 'wb') as f:
        pkl.dump(train_set, f)
        pkl.dump(test_set, f)
        pkl.dump(feature_size, f)


def main():
    df = to_df(RAW_DATA_FILE)
    df, item_cnt, feature_size = remap(df)
    user_df, item_df = gen_user_item_group(df, item_cnt, feature_size)
    # get_stat(user_df, item_df, STAT_PKL_PATH)
    # plt_stat(STAT_PKL_PATH, FIG_PATH)
    gen_dataset(user_df, item_df, item_cnt, feature_size, DATASET_PKL)
    fin = open('../data/taobao/dataset.pkl', 'r')
    trainset = pkl.load(fin)
    testset = pkl.load(fin)
    feature_size = pkl.load(fin)
    fout_crop = open('../data/taobao/dataset_crop.pkl', 'wb')
    fout_fp = open('../data/taobao/dataset_hpmn.pkl', 'wb')
    train_crop = []
    test_crop = []
    train_fp = []
    test_fp = []
    for seq in trainset:
       train_crop.append(crop(seq, 44, 300, 4, 8, 36, 3))
       train_fp.append(front_padding(seq, 300, 4, 36, 3))
    del trainset
    for seq in testset:
       test_crop.append(crop(seq, 44, 300, 4, 8, 36, 3))
       test_fp.append(front_padding(seq, 300, 4, 36, 3))
    pkl.dump(train_crop, fout_crop)
    pkl.dump(test_crop, fout_crop)
    pkl.dump(feature_size, fout_crop)
    pkl.dump(train_fp, fout_fp)
    pkl.dump(test_fp, fout_fp)
    pkl.dump(feature_size, fout_fp)
    fin.close()

if __name__ == '__main__':
    main()
