import time
import torch
import random
import pandas as pd
import numpy as np
from datetime import timedelta
from torch_geometric.data import Data


class LoadData:
    def __init__(self, args):
        # print("prepare data")
        self.graph_path = "datasets/graph"
        self.args = args
        self.nodes = set()

        # node
        edges = []
        edge_weight = []
        with open(f"{self.graph_path}/{args.dataset}.txt", "r") as f:
            for line in f.readlines():
                val = line.split()
                if val[0] not in self.nodes:
                    self.nodes.add(val[0])
                if val[1] not in self.nodes:
                    self.nodes.add(val[1])
                edges.append([int(val[0]), int(val[1])])
                edge_weight.append(float(val[2]))

        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_weight = torch.FloatTensor(edge_weight)

        # feature
        self.nfeat_dim = len(self.nodes)
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = torch.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = torch.FloatTensor(value)
        shape = torch.Size(shape)

        # self.features = th.sparse.FloatTensor(indices, values, shape).to_dense()
        features = torch.sparse.FloatTensor(indices, values, shape)
        self.graph = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)

        # target
        target_fn = f"datasets/text_dataset/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # train val test split
        self.train_lst, self.test_lst = get_train_test(target_fn)

def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst

def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)

def return_seed(nums=10):
    seed = random.sample(range(0, 100000), nums)
    return seed

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))