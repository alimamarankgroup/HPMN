import multiprocessing
import time
import random
import numpy as np


class DataLoader_Mul:
    def __init__(self, dataset, batchsize, max_q_size=10, wait_time=0.1, worker_n=8):
        self.wait_time = wait_time
        self.max_q_size = max_q_size
        self.work = multiprocessing.Queue(maxsize=max_q_size)
        self.results = multiprocessing.Queue(maxsize=max_q_size)
        self.batch_size = batchsize / 2
        self.dataset = open(dataset)
        self.read_stop = multiprocessing.Value('d', 0.0)
        self.work_stop = multiprocessing.Value('d', 0.0)
        self.qsize = multiprocessing.Value('d', 0.0)
        self.work_qsize = multiprocessing.Value('d', 0.0)
        self.worker_n = worker_n
        self.generator_threads = []

        thread = multiprocessing.Process(target=self.producer)
        self.generator_threads.append(thread)
        thread.daemon = True
        thread.start()
        for i in range(worker_n):
            thread = multiprocessing.Process(target=self.worker, args=[i])
            self.generator_threads.append(thread)
            thread.daemon = True
            thread.start()

    def producer(self):
        while self.read_stop.value == 0.0:
            lines = []
            for _ in range(self.batch_size):
                line = self.dataset.readline()
                if not line:
                    self.read_stop.value = 1.0
                    break
                lines.append(line)
            while self.qsize.value >= self.max_q_size:
                time.sleep(self.wait_time)
            if len(lines) != 0:
                self.work.put(lines)
                with self.work_qsize.get_lock():
                    self.work_qsize.value += 1

    def worker(self, name):
        item_cnt = 3269017
        while not (self.read_stop.value == 1. and self.work_qsize.value == 0):
            try:
                lines = self.work.get(timeout=self.wait_time)
            except:
                continue
            with self.work_qsize.get_lock():
                self.work_qsize.value -= 1
            label, item_part, item_part_len, user_part, user_part_len = [], [], [1000 + 1] * 2 * len(lines), [], [184] * 2 * len(lines)
            for line in lines:
                line_items = line.split('\t')
                try:
                    index = int(line_items[0])
                except:
                    print(line_items)
                uid = int(line_items[1])
                item_hist_pos, item_hist_neg = [], []
                for i in map(int, line_items[2].split(',')):
                    item_hist_pos.append([uid + item_cnt, i])
                    item_hist_neg.append([uid + item_cnt, i])
                item_hist_pos.append([uid + item_cnt, int(line_items[3])])
                item_hist_neg.append([uid + item_cnt, int(line_items[4])])

                user_part.append(map(int, line_items[5].split(',')))
                user_part.append(map(int, line_items[6].split(',')))

                t = (index, 1, item_hist_pos)
                label.append(t[1])
                item_part.append(t[2])
                t = (index, 0, item_hist_neg)
                label.append(t[1])
                item_part.append(t[2])
            item_part = np.array(item_part)
            user_part = np.expand_dims(np.array(user_part), 2)
            while self.results.qsize() >= self.max_q_size:
                time.sleep(self.wait_time)
            self.results.put((None, (label, item_part, item_part_len, user_part, user_part_len)))
            with self.qsize.get_lock():
                self.qsize.value += 1.
        with self.work_stop.get_lock():
            self.work_stop.value += 1.

    def __next__(self):
        while self.qsize.value == 0. and self.work_stop.value != self.worker_n:
            time.sleep(self.wait_time)
        if self.qsize.value == 0. and self.work_stop.value == self.worker_n:
            for thread in self.generator_threads:
                thread.terminate()
            raise StopIteration
        re = self.results.get()
        with self.qsize.get_lock():
            self.qsize.value -= 1.
        return re
    
    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class DataLoader_Multi(DataLoader_Mul):
    def producer(self):
        index_pre = 256
        tmp = []
        while self.read_stop.value == 0.0:
            lines = tmp
            tmp = []
            for _ in range(self.batch_size):
                line = self.dataset.readline()
                if not line:
                    self.read_stop.value = 1.0
                    break
                index = int(line.split('\t')[0])
                if index != index_pre:
                    index_pre = index
                    tmp = [line]
                    break
                else:
                    lines.append(line)
            while self.work.qsize() >= self.max_q_size:
                time.sleep(self.wait_time)
            if len(lines) != 0:
                self.work.put(lines)

    def worker(self, name):
        item_cnt = 3269017
        while not (self.read_stop.value == 1. and self.work.empty()):
            if self.work.empty():
                time.sleep(self.wait_time)
                continue
            lines = self.work.get()
            index = None
            uids, label, item_part = [], [], []
            for line in lines:
                line_items = line.split('\t')
                index = int(line_items[0]) - 256
                uid = int(line_items[1])
                uids.append(uid)
                item_hist_pos, item_hist_neg = [], []
                for i in map(int, line_items[2].split(',')):
                    item_hist_pos.append([uid + item_cnt, i])
                    item_hist_neg.append([uid + item_cnt, i])
                item_hist_pos.append([uid + item_cnt, int(line_items[3])])
                item_hist_neg.append([uid + item_cnt, int(line_items[4])])
                t = (index, 1, item_hist_pos)
                label.append(t[1])
                item_part.append(t[2])
                t = (index, 0, item_hist_neg)
                label.append(t[1])
                item_part.append(t[2])
            item_part = np.array(item_part)
            while self.results.qsize() >= self.max_q_size:
                time.sleep(self.wait_time)
            self.results.put((index, uids, label, item_part))
            with self.qsize.get_lock():
                self.qsize.value += 1.
        with self.qsize.get_lock():
            self.work_stop.value += 1.


class DataLoader_unsort(DataLoader_Mul):
    def worker(self, name):
        item_cnt = 3269017
        while not (self.read_stop.value == 1. and self.work_qsize.value == 0):
            try:
                lines = self.work.get(timeout=self.wait_time)
                with self.work_qsize.get_lock():
                    self.work_qsize.value -= 1
            except:
                continue
            index = None
            uids, label, item_part = [], [], []
            for line in lines:
                line_items = line.split('\t')
                index = int(line_items[0])
                uid = int(line_items[1])
                uids.append(uid)
                item_hist_pos, item_hist_neg = [], []
                for i in map(int, line_items[2].split(',')):
                    item_hist_pos.append([uid + item_cnt, i])
                    item_hist_neg.append([uid + item_cnt, i])
                item_hist_pos.append([uid + item_cnt, int(line_items[3])])
                item_hist_neg.append([uid + item_cnt, int(line_items[4])])
                t = (index, 1, item_hist_pos)
                label.append(t[1])
                item_part.append(t[2])
                t = (index, 0, item_hist_neg)
                label.append(t[1])
                item_part.append(t[2])
            item_part = np.array(item_part)
            while self.results.qsize() >= self.max_q_size:
                time.sleep(self.wait_time)
            self.results.put((index, uids, label, item_part))
            with self.qsize.get_lock():
                self.qsize.value += 1.
        with self.work_stop.get_lock():
            self.work_stop.value += 1.


class DataLoader_crop(DataLoader_Mul):
    def __init__(self, dataset, batchsize, max_q_size=10, wait_time=0.1, worker_n=16, user_long=160, user_short=24, item_long=768, item_short=233):
        self.user_long = user_long
        self.user_short = user_short
        self.item_long = item_long
        self.item_short = item_short
        DataLoader_Mul.__init__(self, dataset, batchsize, max_q_size, wait_time, worker_n)

    def worker(self, name):
        item_cnt = 3269017
        while not (self.read_stop.value == 1. and self.work_qsize.value == 0):
            try:
                lines = self.work.get(timeout=self.wait_time)
                with self.work_qsize.get_lock():
                    self.work_qsize.value -= 1
            except:
                continue
            index = None
            uids, label, item_part, user_part = [], [], [], []
            item_long_len, item_short_len, user_long_len, user_short_len = [self.item_long] * 2 * len(lines), [
                self.item_short] * 2 * len(lines), [self.user_long] * 2 * len(lines), [self.user_short] * 2 * len(lines)
            for line in lines:
                line_items = line.split('\t')
                index = int(line_items[0])
                uid = int(line_items[1])
                uids.append(uid)
                item_hist_pos, item_hist_neg = [], []
                for i in map(int, line_items[2].split(',')):
                    item_hist_pos.append([uid + item_cnt, i])
                    item_hist_neg.append([uid + item_cnt, i])
                item_hist_pos.append([uid + item_cnt, int(line_items[3])])
                item_hist_neg.append([uid + item_cnt, int(line_items[4])])

                user_part.append(map(int, line_items[5].split(',')))
                user_part.append(map(int, line_items[6].split(',')))

                t = (index, 1, item_hist_pos)
                label.append(t[1])
                item_part.append(t[2])
                t = (index, 0, item_hist_neg)
                label.append(t[1])
                item_part.append(t[2])
            user_part = np.expand_dims(np.array(user_part), 2)
            item_part = np.array(item_part)
            item_part_long = item_part[:, -self.item_long-self.item_short:-self.item_short, :]
            item_part_short = item_part[:, -self.item_short:, :]
            user_part_long = user_part[:, -self.user_long-self.user_short:-self.user_short, :]
            user_part_short = user_part[:, -self.user_short:, :]
            while self.results.qsize() >= self.max_q_size:
                time.sleep(self.wait_time)
            self.results.put((None, (label, item_part_long, item_part_short, item_long_len, item_short_len, user_part_long,
                              user_part_short, user_long_len, user_short_len)))
            with self.qsize.get_lock():
                self.qsize.value += 1.
        with self.work_stop.get_lock():
            self.work_stop.value += 1.


class DataLoader:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_of_step = len(dataset) // self.batch_size
        if self.batch_size * self.num_of_step < len(dataset):
            self.num_of_step += 1
        self.i = 0  # current position in dataset

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.num_of_step:
            raise StopIteration

        ts = self.dataset[self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
        label, item_part, item_part_len, user_part, user_part_len = [], [], [], [], []

        for t in ts:
            label.append(t[0])
            # shuf = np.random.shuffle(t[1][0:t[2]-1])
            # shuf = np.append(shuf, t[1][t[2]-1])
            # item_part.append(shuf)
            item_part.append(t[1])
            item_part_len.append(t[2])
            user_part.append(t[3])
            user_part_len.append(t[4])
        item_part = np.array(item_part)
        user_part = np.array(user_part)
        self.i += 1
        return self.i, (label, item_part, item_part_len, user_part, user_part_len)

    def next(self):
        return self.__next__()

class RUMDataloader():
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_of_step = len(dataset) // self.batch_size
        if self.batch_size * self.num_of_step < len(dataset):
            self.num_of_step += 1
        self.i = 0  # current position in dataset

    def __iter__(self):
        return self

    def next(self):
        if self.i == self.num_of_step:
            random.shuffle(self.dataset)
            self.i = 0

        ts = self.dataset[self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
        label, item_part, item_part_len, user_part, user_part_len = [], [], [], [], []

        for t in ts:
            label.append(t[0])
            item_part.append(t[1])
            item_part_len.append(t[2])
            user_part.append(t[3])
            user_part_len.append(t[4])
        item_part = np.array(item_part)
        user_part = np.array(user_part)
        self.i += 1
        item_back = []
        users = []
        items = []
        cates = []
        users_back = []
        items_back = []
        cates_back = []

        for current_i in range(len(item_part[0])):
            user_batch = []
            item_batch = []
            cate_batch = []
            for j in range(len(ts)):
                if current_i < item_part_len[j] - 1:
                    user_batch.append(item_part[j][current_i][0])
                    item_batch.append(item_part[j][current_i][1])
                    cate_batch.append(item_part[j][current_i][2])
                elif current_i == item_part_len[j] - 1:
                    users_back.append(item_part[j][current_i][0])
                    items_back.append(item_part[j][current_i][1])
                    cates_back.append(item_part[j][current_i][2])
            if len(user_batch) > 0:
                users.append(user_batch)
                items.append(item_batch)
                cates.append(cate_batch)
        # return self.i, (label, item_part, item_part_len, user_part, user_part_len)
        # print users, items, cates, users_back, items_back, cates_back, label
        return self.i, users, items, cates, users_back, items_back, cates_back, label


class CroppedLoader(DataLoader):
    def next(self):
        if self.i == self.num_of_step - 1:
            raise StopIteration

        ts = self.dataset[self.i * self.batch_size: min(len(self.dataset), (self.i + 1) * self.batch_size)]
        label, user_mem, user_que, user_mem_len, user_que_len, item_mem, item_que, item_mem_len, item_que_len = [], [], [], [], [], [], [], [], []
        for t in ts:
            label.append(t[0])
            user_mem.append(t[1])

            user_que.append(t[2])
            # shuf = np.random.shuffle(t[2][0:t[4]-1])
            # shuf = n  p.append(shuf, t[2][t[4]-1])
            # user_que.append(shuf)

            user_mem_len.append(t[3])
            user_que_len.append(t[4])
            item_mem.append(t[5])
            item_que.append(t[6])
            item_mem_len.append(t[7])
            item_que_len.append(t[8])

        #label = np.array(label, dtype='int32')
        user_mem = np.array(user_mem, dtype='int32')
        user_que = np.array(user_que, dtype='int32')
        user_mem_len = np.array(user_mem_len, dtype='int32')
        user_que_len = np.array(user_que_len, dtype='int32')
        item_mem = np.array(item_mem, dtype='int32')
        item_que = np.array(item_que, dtype='int32')
        item_mem_len = np.array(item_mem_len, dtype='int32')
        item_que_len = np.array(item_que_len, dtype='int32')

        self.i += 1

        return self.i, (
            label, user_mem, user_que, user_mem_len, user_que_len, item_mem, item_que, item_mem_len, item_que_len)
