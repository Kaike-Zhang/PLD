from collections import defaultdict
import csv
import datetime
import json
import os
import random
from typing import Counter
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


class BasicDataset(Dataset):
    def __init__(self, path, config) -> None:
        super().__init__()
        self.path = path
        self.config = config
        self.min_len = config["min_interaction"]

        self._load_data()

    def __len__(self):
        return self.n_users
    
    def __getitem__(self, index):
        return index
    
    def _load_org_data(self):
        '''
        data format:
        User ID; [Item IDs]
        '''
        # 读取 JSON 文件
        with open(os.path.join(self.path, "data.json"), 'r') as f:
            users = json.load(f)


        valid_users = {user: items for user, items in users.items() if len(items) >= self.config['min_interaction']}

        user_mapping = {user: idx for idx, user in enumerate(valid_users.keys())}
        item_set = {item for items in valid_users.values() for item in items}
        item_mapping = {item: idx for idx, item in enumerate(item_set)}

        user_item_dict = {user_mapping[user]: [item_mapping[item] for item in items] for user, items in valid_users.items()}
        
        
        if not os.path.exists(os.path.join(self.path, "processed")): os.makedirs(f'{self.path}/processed', exist_ok=True)
        with open(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle"), 'wb') as f:
            pickle.dump((user_item_dict, len(user_mapping), len(item_mapping), item_mapping), f)
        
        return user_item_dict, len(user_mapping), len(item_mapping), item_mapping


    def _load_data(self):
        if os.path.exists(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle")):
            with open(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle"), 'rb') as f:
                self.user_interactions, self.n_users, self.n_items, item_map = pickle.load(f)
        else:
            self.user_interactions, self.n_users, self.n_items, item_map = self._load_org_data()

    def get_train_batch(self, idx):
        raise NotImplementedError

    def get_val_batch(self, idx):
        raise NotImplementedError

    def get_test_batch(self, idx):
        raise NotImplementedError

    @staticmethod
    def collate_fn(samples):
        return samples


class CFDataset(BasicDataset):
    def __init__(self, path, config) -> None:
        super().__init__(path, config)
        self.split_ratio = [0.7, 0.1, 0.2]
        self._build_set()
    
    def __len__(self):
        return self.n_train_num
    
    def _build_set(self):
        self.n_train_num = 0
        self.train_data = [[] for _ in range(self.n_users)]
        self.val_data = [[] for _ in range(self.n_users)]
        self.test_data = [[] for _ in range(self.n_users)]

        all_num = 0

        for user in range(self.n_users):
            random.shuffle(self.user_interactions[user])
            n_inter_items = len(self.user_interactions[user])
            n_train_items = int(n_inter_items * self.split_ratio[0])
            n_test_items = int(n_inter_items * self.split_ratio[2])
            self.train_data[user] += self.user_interactions[user][:n_train_items]
            self.val_data[user] += self.user_interactions[user][n_train_items:-n_test_items]
            self.test_data[user] += self.user_interactions[user][-n_test_items:]
            self.n_train_num += n_train_items
            all_num += n_inter_items
        
        self.avg_inter = int(self.n_train_num / self.n_users)

        self._add_noise(self.config["noise"], self.config["add_p"])


        print(f"#User: {self.n_users}, #Item: {self.n_items}, #Ratings: {all_num}, AvgLen: {int(10 * (all_num / self.n_users)) / 10}, Sparsity: {100 - int(10000 * all_num / (self.n_users * self.n_items)) / 100}")

    def _add_noise(self, noise_ratio, add_p):
        user_noise = {}
        if noise_ratio == 0.0:
            return
        for user, interaction in enumerate(self.train_data):
            noisy = int(noise_ratio * (len(interaction)))
            # TODO: 从 range(0, self.n_items) 随机选取 noisy 个不在 interaction 中的 item，拼接到 interaction 后
            if noisy > 0:
                available_items = set(range(self.n_items)) - set(interaction)
                noisy_items = random.sample(available_items, noisy)
                
                user_noise[user] = noisy_items

                # 拼接到 interaction 后
                interaction.extend(noisy_items)
                self.train_data[user] = interaction



    def get_train_batch(self, inter_list, multi_sample=False, k=5):
        inter_list = inter_list.squeeze().tolist()
        pos_item_list = []
        neg_item_list = []
        user_list = np.random.randint(0, self.n_users, len(inter_list))
        
        if multi_sample:
            for user in user_list:
                if len(self.train_data[user]) >= k:
                    pos_items = np.random.choice(self.train_data[user], k, replace=False)
                else:
                    pos_items = np.random.choice(self.train_data[user], k, replace=True)
                pos_item_list.append(pos_items)
                neg_item = random.randint(0, self.n_items-1)
                while neg_item in self.train_data[user]:
                    neg_item = random.randint(0, self.n_items-1)
                neg_item_list.append(neg_item)
        else:
            for user in user_list:
                pos_item_list.append(np.random.choice(self.train_data[user]))
                neg_item = random.randint(0, self.n_items-1)
                while neg_item in self.train_data[user]:
                    neg_item = random.randint(0, self.n_items-1)
                neg_item_list.append(neg_item)
        return user_list, np.array(pos_item_list), np.array(neg_item_list)

    def get_val_batch(self, user_list):
        return np.array(user_list), [self.val_data[user] for user in user_list], [self.train_data[user] for user in user_list]
    
    def get_test_batch(self, user_list):
        return np.array(user_list), [self.test_data[user] for user in user_list], [self.train_data[user] + self.val_data[user] for user in user_list]

    def gcn_graph(self):
        user_list = []
        item_list = []
        for user in range(self.n_users):
            items = self.train_data[user]
            for item in items:
                user_list.append(user)
                item_list.append(item)
        user_dim = torch.LongTensor(user_list)
        item_dim = torch.LongTensor(item_list)
        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()

        Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.n_items, self.n_users+self.n_items]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)

        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.n_items, self.n_users+self.n_items]))
        Graph = Graph.coalesce()

        return Graph