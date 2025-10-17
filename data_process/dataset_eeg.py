from data_process.partition import *
from utils.math_utils import mapminmax
from torch.utils.data import Dataset as Dataset_nn
import scipy.io as sio
import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, data, target, data_idxs=None,
                 transform=None, target_transform=None):
        self.data_idxs = data_idxs
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__(data, target)

    def __build_truncated_dataset__(self, data_para, target_para):
        data_return = data_para
        target_return = target_para
        if self.data_idxs is not None:
            data_return = data_para[self.data_idxs]
            target_return = target_para[self.data_idxs]
        return data_return, target_return

    def __getitem__(self, index):
        data_return, target = self.data[index], self.target[index]

        if self.transform is not None:
            data_return = self.transform(data_return)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data_return, target

    def __len__(self):
        return len(self.data)


def get_dataset_mat(dir_dataset, p_args):
    load_data = sio.loadmat(dir_dataset)

    inputs = load_data['inputs'].astype(np.float32)

    # Normalize if requested
    if p_args.b_norm_dataset:
        inputs = mapminmax(inputs)

    # Add noise if requested
    if p_args.nl > 0.0:
        element_num = inputs.shape[0] * inputs.shape[1]
        noise_num = int(p_args.nl * element_num)
        mu, sigma = 0, 0.8
        noise = np.random.normal(mu, sigma, element_num).reshape(inputs.shape)

        mask = np.zeros((element_num, 1))
        mask[0:noise_num, :] = 1
        mask = mask[np.random.permutation(element_num), :].reshape(inputs.shape)
        mask = mask == 1
        inputs[mask] = noise[mask] + inputs[mask]

    # ✅ Fix target handling
    targets = load_data['targets'].astype(np.int64).squeeze()
    targets = targets - targets.min()
    n_class = int(targets.max() + 1)

    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)  # ensure shape (N,1)

    task = str(load_data['task'])

    # Partition
    partition_strategy = FedKfoldPartition(p_args)
    partition_strategy.partition(targets, True, 0)

    dataset = Dataset(inputs, targets)
    return dataset, n_class, task
