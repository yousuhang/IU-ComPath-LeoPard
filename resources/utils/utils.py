import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_MIL_reg(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    time = torch.FloatTensor([item[2] for item in batch])
    return [img, label, time]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset),
                        collate_fn=collate_MIL, **kwargs)
    return loader


def get_split_loader(split_dataset, training=False, testing=False, weighted=False, collate_func=collate_MIL):
    """
        return either the validation loader or training loader
    """
    kwargs = {'num_workers': 2} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)),
                                    collate_fn=collate_func, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate_func, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset),
                                collate_fn=collate_func, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset) * 0.1), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate_func,
                            **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_folds(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    if label_frac != 1:
        return NotImplementedError
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    # np.random.shuffle(indices)
    all_val_ids = []
    all_test_ids = []
    sampled_train_ids = []
    if custom_test_ids is not None:  # pre-built test split, do not need to sample
        all_test_ids.extend(custom_test_ids)
    else:
        raise ValueError('Please Predefine a fixed custom test with indeces!')
    val_ids_folded = {}
    for c in range(len(val_num)):
        indeces_candidate = indices.copy()

        for i in range(n_splits):

            # print(i, c)
            # print(len(indeces_candidate))
            possible_indices = np.intersect1d(cls_ids[c], indeces_candidate)  #all indices of this class
            if i < n_splits - 1:
                fold_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids
                # fold_ids = possible_indices[:val_num[c]]  # validation ids

                fold_ids = fold_ids.tolist()
                # print(fold_ids)
                indeces_candidate = np.setdiff1d(possible_indices.copy(),
                                                 fold_ids)  # indices of this class left after validation
                # print(len(indeces_candidate))

            else:
                fold_ids = possible_indices.tolist()

            if c == 0:
                print(i)
                val_ids_folded.update({f'{i}': fold_ids})
                print(val_ids_folded.keys())
            else:
                print('extend', fold_ids)
                val_ids_folded[f'{i}'].extend(fold_ids)
    train_ids_folded = {}
    for i in range(n_splits):
        print(i, val_ids_folded[f'{i}'])
        train_ids_folded.update({f'{i}': np.setdiff1d(indices, val_ids_folded[f'{i}']).tolist()})

    return sort_dict(val_ids_folded), sort_dict(train_ids_folded), sorted(all_test_ids)

def generate_split(val_ids_folded, train_ids_folded, all_test_ids,n_splits=5):
    for i in range(n_splits):
        yield train_ids_folded[f'{i}'], val_ids_folded[f'{i}'],all_test_ids

    # remaining_ids = []
    # if label_frac == 1:
    #     sampled_train_ids.extend(remaining_ids)


        # sample_num  = math.ceil(len(remaining_ids) * label_frac)
        # slice_ids = np.arange(sample_num)
        # sampled_train_ids.extend(remaining_ids[slice_ids])


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def sort_dict(dictionary):
    for k, v in dictionary.items():
        dictionary[k] = sorted(dictionary[k])
    return dictionary