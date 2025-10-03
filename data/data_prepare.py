import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from collections import defaultdict
import os
import random



class UnifiedDataset(Dataset):
    def __init__(self, data_tensor):
        """
        Args:
            data_tensor: shape [num_entries, 10, max_label_size]
                        结构说明:
                        [0,:]: user_idx (直接取[0,0]的值)
                        [1,:]: item_idx (直接取[1,0]的值)
                        [2,:]: rating (直接取[2,0]的原始值)
                        [3-9,:]: 7个attributes的one-hot编码
        """
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # 直接取user_idx、item_idx和rating的原始值
        user_idx = entry[0, 0].item()  # 取[0,0]位置的值
        item_idx = entry[1, 0].item()  # 取[1,0]位置的值
        rating = entry[2, 0].item()  # 取[2,0]位置的rating原始值
        # 提取7个attributes的one-hot labels
        attributes = entry[3:10]  # shape [7, max_label_size]

        return (
            torch.tensor([user_idx, item_idx, rating], dtype=torch.float32),  # [3]
            attributes  # [7, max_label_size]
        )


#==============generate dataloader for group-item data===============
class GroupOpinionDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def collate_fn(batch):
    # 自动将 batch 组织成 dict of lists/tensors
    return {
        'user_idx': torch.tensor([b['user_idx'] for b in batch], dtype=torch.long),
        'item_idx': torch.tensor([b['item_idx'] for b in batch], dtype=torch.long),
        'rating': torch.tensor([b['rating'] for b in batch], dtype=torch.float),
        'user_opinion_label': torch.stack([b['user_opinion_label'] for b in batch]),  # [B, 7, 4]
        'group_user_ids': [b['group_user_ids'] for b in batch],                       # List[List[int]]
        'group_labels': [b['group_labels'] for b in batch],                           # List[Tensor[G, 7, 4]]
        'group_sum_label': torch.stack([b['group_sum_label'] for b in batch]),
    }
def load_indices(file):
    with open(file, 'r') as f:
        indices = f.read().strip().split(' ')
        return [int(i) for i in indices]

def load_group_data_save_index(data, batch_size, seed=42):
    random.seed(seed)
    # # data = torch.load(path)

    # user_to_items = defaultdict(list)
    # for idx, sample in enumerate(data):
    #     sample['global_idx'] = idx  # 新增: 给每个样本添加其全局 index
    #     u = sample['user_idx']
    #     user_to_items[u].append(sample)
    #
    # train_data, val_data, test_data = [], [], []
    #
    # train_indices, val_indices, test_indices = [], [], []  # 新增: 记录索引
    #
    # for u, samples in user_to_items.items():
    #     random.shuffle(samples)
    #     n = len(samples)
    #     if n == 1:
    #         train_data.append(samples[0])
    #         val_data.append(samples[0])
    #         test_data.append(samples[0])
    #         train_indices.append(str(samples[0]['global_idx']))
    #         val_indices.append(str(samples[0]['global_idx']))
    #         # test_indices.append(str(samples[0]['global_idx']))
    #     elif n == 2:
    #         train_data.append(samples[0])
    #         val_data.append(samples[1])
    #         test_data.append(samples[1])
    #         train_indices.append(str(samples[0]['global_idx']))
    #         val_indices.append(str(samples[1]['global_idx']))
    #         # test_indices.append(str(samples[1]['global_idx']))
    #     elif n == 3:
    #         train_data.append(samples[0])
    #         val_data.append(samples[1])
    #         test_data.append(samples[2])
    #         train_indices.append(str(samples[0]['global_idx']))
    #         val_indices.append(str(samples[1]['global_idx']))
    #         test_indices.append(str(samples[2]['global_idx']))
    #     elif n > 3 and n < 10:
    #         # n_train = max(1, int(n * 0.9)) #0.8/0.15
    #         # n_val = int(n * 0.15)
    #         # train_data += samples[:n_train]
    #         # val_data += samples[n_train:n_val]
    #         # test_data += samples[n_val:]
    #
    #         n_train = max(1, int(round(n * 0.8)))
    #         rem = n - n_train
    #         n_val = rem // 2
    #         n_test = rem - n_val
    #
    #         train_data += samples[:n_train]
    #         val_data += samples[n_train: n_train + n_val]
    #         test_data += samples[n_train + n_val: n_train + n_val + n_test]
    #
    #         train_indices += [str(s['global_idx']) for s in samples[:n_train]]
    #         val_indices += [str(s['global_idx']) for s in samples[n_train: n_train + n_val]]
    #         test_indices += [str(s['global_idx']) for s in samples[n_train + n_val: n_train + n_val + n_test]]
    #
    #         # train_indices += [str(s['global_idx']) for s in samples[:n_train]]
    #         # val_indices += [str(s['global_idx']) for s in samples[n_train:n_val]]
    #         # test_indices += [str(s['global_idx']) for s in samples[n_val:]]
    #     else:
    #         n_train = int(round(n * 0.8))
    #         rem = n - n_train
    #         n_val = rem // 2
    #         n_test = rem - n_val
    #
    #         train_data += samples[:n_train]
    #         val_data += samples[n_train: n_train + n_val]
    #         test_data += samples[n_train + n_val: n_train + n_val + n_test]
    #
    #         train_indices += [str(s['global_idx']) for s in samples[:n_train]]
    #         val_indices += [str(s['global_idx']) for s in samples[n_train: n_train + n_val]]
    #         test_indices += [str(s['global_idx']) for s in samples[n_train + n_val: n_train + n_val + n_test]]
    #
    # print(f"📦 Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    #
    # # 保存 index 文件
    # with open('data/Sports/train_indices.index', 'w') as f:
    #     f.write(' '.join(train_indices))
    # with open('data/Sports/val_indices.index', 'w') as f:
    #     f.write(' '.join(val_indices))
    # with open('data/Sports/test_indices.index', 'w') as f:
    #     f.write(' '.join(test_indices))

  #=====================use indices to avoid random split==========================
    full_dataset = GroupOpinionDataset(data)
    #
    # # 加载 train, val, test indices
    train_indices = load_indices('data/beauty/train_indices.index')
    val_indices = load_indices('data/beauty/val_indices.index')
    test_indices = load_indices('data/beauty/test_indices.index')
    #yelp
    # train_indices = load_indices('data/ablation_yelp/train_indices.index')
    # val_indices = load_indices('data/ablation_yelp/val_indices.index')
    # test_indices = load_indices('data/ablation_yelp/test_indices.index')
    # train_indices = load_indices('data/train_indices.index')
    # val_indices = load_indices('data/val_indices.index')
    # test_indices = load_indices('data/test_indices.index')
    # #
    # # # # 构造 Subset dataset
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# #======================================================
#     train_loader = DataLoader(GroupOpinionDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(GroupOpinionDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(GroupOpinionDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    return train_loader, val_loader, test_loader

def load_group_data(data, batch_size, seed=42):
    random.seed(seed)
    # data = torch.load(path)

    # 用户 u → 该用户相关的数据项列表
    user_to_items = defaultdict(list)
    for sample in data:
        u = sample['user_idx']
        user_to_items[u].append(sample)

    train_data, val_data, test_data = [], [], []

    for u, samples in user_to_items.items():
        random.shuffle(samples)
        n = len(samples)
        if n == 1:
            train_data.append(samples[0])
            val_data.append(samples[0])
            test_data.append(samples[0])
        elif n == 2:
            train_data.append(samples[0])
            val_data.append(samples[1])
            test_data.append(samples[1])
        elif n == 3:
            train_data.append(samples[0])
            val_data.append(samples[1])
            test_data.append(samples[2])
        elif n > 3 and n < 10:
            n_train = max(1, int(n * 0.8))
            # n_val = n_train + 2
            n_val = int(n * 0.15)
            train_data += samples[:n_train]
            val_data += samples[n_train:n_val]
            test_data += samples[n_val:]
        else:
            n_train = int(n * 0.8)
            n_val = int(n * 0.15)
            train_data += samples[:n_train]
            val_data += samples[n_train:n_train + n_val]
            test_data += samples[n_train + n_val:]


    print(f"📦 Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

    train_loader = DataLoader(GroupOpinionDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(GroupOpinionDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(GroupOpinionDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
