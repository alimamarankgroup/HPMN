import sys
if sys.version_info < (3, 0):
    import cPickle as pkl
else:
    import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime
from util import front_padding, crop

random.seed(1111)

REVIEW_FILE = '../data/raw_data/amazon/Electronics_5.json'
META_FILE = '../data/raw_data/amazon/meta_Electronics.json'

SAVE_PKL_PATH = '../data/raw_data/amazon/save.pkl'
DATASET_PKL_PATH = '../data/amazon/dataset.pkl'

FIG_PATH = '../../data/amazon/Electronics/Electronics_Stat.png'
MAX_LEN = 100

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

def aggregator(f_review, f_meta):
    # get dataframe from raw data file
    review_df = to_df(f_review)[['reviewerID', 'asin', 'unixReviewTime']]
    meta_df = to_df(f_meta)[['asin', 'categories']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
    meta_df = meta_df.drop_duplicates(subset=['asin'])
    print("raw file to dataframe completed")

    # join asin with its relative category id
    item_list = meta_df['asin'].tolist()
    cate_list = meta_df['categories'].tolist()
    item_cate_dict = dict(zip(item_list, cate_list))

    review_df['category'] = review_df['asin'].map(lambda x: item_cate_dict[x])
    print("join item with its categories completed")
    return review_df

# map to one-hot index
def remap(df):
    asin_key = sorted(df['asin'].unique().tolist())
    asin_len = len(asin_key)
    asin_map = dict(zip(asin_key, range(asin_len)))
    df['asin'] = df['asin'].map(lambda x: asin_map[x])

    cate_key = sorted(df['category'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(asin_len, asin_len + cate_len)))
    df['category'] = df['category'].map(lambda x: cate_map[x])

    user_key = sorted(df['reviewerID'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(asin_len + cate_len, asin_len + cate_len + user_len)))
    df['reviewerID'] = df['reviewerID'].map(lambda x: user_map[x])
    
    return df, asin_len, asin_len + cate_len + user_len #remapped df and feature size

def plt_stat(save_pkl, fig_path):
    with open(save_pkl, 'rb') as f:
        user_df = pickle.load(f)
        item_df = pickle.load(f)
    
    # plot statistic
    user_hist_stat = []
    item_hist_stat = []

    for uid, hist in user_df:
        length = len(hist['asin'].tolist())
        user_hist_stat.append(length)
    for iid, hist in item_df:
        length = len(hist['reviewerID'].tolist())
        item_hist_stat.append(length)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    edges_user = range(1, 100, 1)
    plt.hist(user_hist_stat, bins = edges_user)
    plt.xlabel('history length')
    plt.ylabel('user cnt')
    plt.title('user history statistics')

    plt.subplot(1, 2, 2)
    edges_item = range(1, 100, 1)
    plt.hist(item_hist_stat, bins = edges_item)
    plt.xlabel('history length')
    plt.ylabel('item cnt')
    plt.title('item history statistics')

    plt.savefig(fig_path)

def gen_user_item_group(df, item_cnt, feature_size):
    # get two sub dataframe: groupby user and item
    user_df = df.sort_values(['reviewerID', 'unixReviewTime']).groupby('reviewerID')
    item_df = df.sort_values(['asin', 'unixReviewTime']).groupby('asin')
    with open(SAVE_PKL_PATH, 'wb') as f:
        pickle.dump(user_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(item_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(item_cnt, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(feature_size, f, pickle.HIGHEST_PROTOCOL)
    print("group completed")
    return user_df, item_df

def gen_dataset(save_pkl, dataset_pkl):
    with open(save_pkl, 'rb') as f:
        user_df = pickle.load(f)
        item_df = pickle.load(f)
        item_cnt = pickle.load(f)
        feature_size = pickle.load(f)
    print("file load completed")

    train_set = []
    test_set = []

    # get each user's last touch point time
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['unixReviewTime'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    # all_delta_time = []

    for uid, hist in user_df:
        item_hist = hist['asin'].tolist()
        cate_hist = hist['category'].tolist()
        target_item_time = hist['unixReviewTime'].tolist()[-1]

        # get time delta between user's history items
        # delta_time_item = (hist['unixReviewTime'] - hist['unixReviewTime'].shift(1)).fillna(0).tolist()
        # delta_time_item = [(datetime.fromtimestamp(t) - datetime.fromtimestamp(0)).days + feature_size for t in delta_time_item]
        # all_delta_time += delta_time_item

        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        label = 1
        test = (target_item_time > split_time)
        # neg sampling
        neg = random.randint(0, 1)
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]:
                target_item = random.randint(0, item_cnt - 1)
                target_item_cate = item_df.get_group(target_item)['category'].tolist()[0]

        # the item history part of the sample
        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i]])
        item_part.append([uid, target_item, target_item_cate])
        item_part_len = min(len(item_part), MAX_LEN)

        # choose the item side information: which user has clicked the target item
        item_side = item_df.get_group(target_item)
        user_hist = item_side['reviewerID'].tolist()
        user_hist_time = item_side['unixReviewTime'].tolist()

        # delta_time_user = (item_side['unixReviewTime'] - item_side['unixReviewTime'].shift(1)).fillna(0).tolist()
        # delta_time_user = [(datetime.fromtimestamp(t) - datetime.fromtimestamp(0)).days + feature_size for t in delta_time_user]
        # all_delta_time += delta_time_user

        # the user history part of the sample
        user_part = []
        for i in range(len(user_hist)):
            if user_hist_time[i] < target_item_time:
                user_part.append([target_item, user_hist[i]])
        user_part.append([target_item, uid])

        user_part_len = min(len(user_part), MAX_LEN)

        if len(user_part) == 0:
            continue

        # padding history with 0
        if len(item_part) <= MAX_LEN:
            item_part_pad = item_part + [[0] * 3] * (MAX_LEN - len(item_part))
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN:len(item_part)]
        if len(user_part) <= MAX_LEN:
            user_part_pad = user_part + [[0] * 2] * (MAX_LEN - len(user_part))
        else:
            user_part_pad = user_part[len(user_part) - MAX_LEN:len(user_part)]

        # gen sample
        sample = (label, item_part_pad, item_part_len, user_part_pad, user_part_len)

        if test:
            test_set.append(sample)
        else:
            train_set.append(sample)

    random.shuffle(train_set)
    random.shuffle(test_set)
    # feature_size += (max(all_delta_time) - feature_size) + 1

    with open(dataset_pkl, 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(feature_size, f, pickle.HIGHEST_PROTOCOL)


def gen_dataset_dien(save_pkl, dataset_pkl):
    with open(save_pkl, 'rb') as f:
        user_df = pickle.load(f)
        item_df = pickle.load(f)
        item_cnt = pickle.load(f)
        feature_size = pickle.load(f)
    print("file load completed")

    train_set = []
    test_set = []

    for uid, hist in user_df:
        item_hist = hist['asin'].tolist()
        cate_hist = hist['category'].tolist()
        target_item_time = hist['unixReviewTime'].tolist()[-1]

        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        
        neg_target_item = target_item
        neg_target_cate = target_item_cate
        while neg_target_item == target_item:
            neg_target_item = random.randint(0, item_cnt - 1)
            neg_target_cate = item_df.get_group(neg_target_item)['category'].tolist()[0]
        
        # the item history part of the sample
        item_part_pos = []
        for i in range(len(item_hist) - 1):
            item_part_pos.append([uid, item_hist[i], cate_hist[i]])
        item_part_pos.append([uid, target_item, target_item_cate])

        item_part_neg = []
        for i in range(len(item_hist) - 1):
            item_part_neg.append([uid, item_hist[i], cate_hist[i]])
        item_part_neg.append([uid, neg_target_item, neg_target_cate])
        
        item_part_len = min(len(item_part_pos), MAX_LEN)

        # choose the item side information: which user has clicked the target item
        item_side = item_df.get_group(target_item)
        user_hist = item_side['reviewerID'].tolist()
        user_hist_time = item_side['unixReviewTime'].tolist()

        # the user history part of the sample
        user_part_pos = []
        for i in range(len(user_hist)):
            if user_hist_time[i] < target_item_time:
                user_part_pos.append([target_item, user_hist[i]])
        user_part_pos.append([target_item, uid])

        user_part_neg = []
        for i in range(len(user_hist)):
            if user_hist_time[i] < target_item_time:
                user_part_neg.append([target_item, user_hist[i]])
        user_part_neg.append([neg_target_item, uid])        

        user_part_len = min(len(user_part_pos), MAX_LEN)

        if user_part_len == 0:
            continue

        # padding history with 0
        if len(item_part_pos) <= MAX_LEN:
            item_part_pos_pad = item_part_pos + [[0] * 3] * (MAX_LEN - len(item_part_pos))
            item_part_neg_pad = item_part_neg + [[0] * 3] * (MAX_LEN - len(item_part_neg))
        else:
            item_part_pos_pad = item_part_pos[len(item_part_pos) - MAX_LEN:len(item_part_pos)]
            item_part_neg_pad = item_part_neg[len(item_part_neg) - MAX_LEN:len(item_part_neg)]
        if len(user_part_pos) <= MAX_LEN:
            user_part_pos_pad = user_part_pos + [[0] * 2] * (MAX_LEN - len(user_part_pos))
            user_part_neg_pad = user_part_neg + [[0] * 2] * (MAX_LEN - len(user_part_neg))
        else:
            user_part_pos_pad = user_part_pos[len(user_part_pos) - MAX_LEN:len(user_part_pos)]
            user_part_neg_pad = user_part_neg[len(user_part_neg) - MAX_LEN:len(user_part_neg)]
            

        # gen sample
        sample_pos = (1, item_part_pos_pad, item_part_len, user_part_pos_pad, user_part_len)
        sample_neg = (0, item_part_neg_pad, item_part_len, user_part_neg_pad, user_part_len)

        rand_int = random.randint(1, 10)
        if rand_int == 2:
            test_set.append(sample_pos)
            test_set.append(sample_neg)
        else:
            train_set.append(sample_pos)
            train_set.append(sample_neg)

    # random.shuffle(train_set)
    # random.shuffle(test_set)
    # feature_size += (max(all_delta_time) - feature_size) + 1

    with open(dataset_pkl, 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(feature_size, f, pickle.HIGHEST_PROTOCOL)

def main():
    df = aggregator(REVIEW_FILE, META_FILE)
    df, item_cnt, feature_size = remap(df)
    user_df, item_df = gen_user_item_group(df, item_cnt, feature_size)

    # plt_stat(SAVE_PKL_PATH, FIG_PATH)
    gen_dataset(SAVE_PKL_PATH, DATASET_PKL_PATH)
    fin = open('../data/amazon/dataset.pkl', 'r')
    trainset = pkl.load(fin)
    testset = pkl.load(fin)
    feature_size = pkl.load(fin)
    fout_crop = open('../data/amazon/dataset_crop.pkl', 'wb')
    fout_fp = open('../data/amazon/dataset_hpmn.pkl', 'wb')
    train_crop = []
    test_crop = []
    train_fp = []
    test_fp = []
    for seq in trainset:
       train_crop.append(crop(seq, 10, 100, 3, 10, 100, 2))
       train_fp.append(front_padding(seq, 100, 3, 100, 2))
    del trainset
    for seq in testset:
       test_crop.append(crop(seq, 10, 100, 3, 10, 100, 2))
       test_fp.append(front_padding(seq, 100, 3, 100, 2))
    pkl.dump(train_crop, fout_crop)
    pkl.dump(test_crop, fout_crop)
    pkl.dump(feature_size, fout_crop)
    pkl.dump(train_fp, fout_fp)
    pkl.dump(test_fp, fout_fp)
    pkl.dump(feature_size, fout_fp)
    fin.close()
    fout_crop.close()
    fout_fp.close()

if __name__ == '__main__':
    main()

