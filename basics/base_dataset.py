import torch
from utils.hparams import hparams
import numpy as np
import os

class BaseDataset(torch.utils.data.Dataset):
    '''
        Base class for datasets.
        1. *ordered_indices*:
            if self.shuffle == True, shuffle the indices;
            if self.sort_by_len == True, sort data by length;
        2. *sizes*:
            clipped length if "max_frames" is set;
        3. *num_tokens*:
            unclipped length.

        Subclasses should define:
        1. *collate*:
            take the longest data, pad other data to the same length;
        2. *__getitem__*:
            the index function.
    '''
    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        size = min(self._sizes[index], hparams['max_frames'])
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
                # 先random, 然后稳定排序, 保证排序后同长度的数据顺序是依照random permutation的 (被其随机打乱).
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))
