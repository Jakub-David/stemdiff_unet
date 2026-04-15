import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
from dataset_enhancement import *


class STEMDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.input_h5 = self.dataset_dir / "x_input.h5"
        self.target_h5 = self.dataset_dir / "x_target.h5"

        self.index_map = []

        with h5py.File(self.input_h5, 'r') as f_in, \
             h5py.File(self.target_h5, 'r') as f_tar:
            in_keys = sorted(f_in.keys())
            tar_keys = sorted(f_tar.keys())

            for in_k, tar_k in zip(in_keys, tar_keys):
                in_len = f_in[in_k].shape[0]
                tar_len = f_tar[tar_k].shape[0]

                if in_len != tar_len:
                    raise ValueError(f"Length mismatch at {in_k} ({in_len}) and {tar_k} ({tar_len})")

                for i in range(in_len):
                    self.index_map.append((in_k, tar_k, i))
        
        # We'll open the files lazily in __getitem__
        self.in_fh = None
        self.tar_fh = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # This part ensures each Worker Process gets its own file handle
        if self.in_fh is None:
            self.in_fh = h5py.File(self.input_h5, 'r')
            self.tar_fh = h5py.File(self.target_h5, 'r')

        in_key, tar_key, img_idx = self.index_map[idx]

        x = self.in_fh[in_key][img_idx]
        y = self.tar_fh[tar_key][img_idx]

        if x.ndim == 2:
            x = x[None, ...]
        if y.ndim == 2:
            y = y[None, ...]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y
    

class AugmentedDataset(Dataset):
    def __init__(self, dataset_path):
        self.input_h5 = dataset_path

        self.index_map = []

        with h5py.File(self.input_h5, 'r') as f_in:
            in_keys = sorted(f_in.keys())

            for in_k in in_keys:
                in_len = f_in[in_k].shape[0]

                for i in range(in_len):
                    self.index_map.append((in_k, i))
        
        # We'll open the files lazily in __getitem__
        self.in_fh = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # This part ensures each Worker Process gets its own file handle
        if self.in_fh is None:
            self.in_fh = h5py.File(self.input_h5, 'r')

        in_key, img_idx = self.index_map[idx]

        x = self.in_fh[in_key][img_idx]
        y = self.augment(x)

        if x.ndim == 2:
            x = x[None, ...]
        if y.ndim == 2:
            y = y[None, ...]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y
    
    def augment(self, x):
        ker_size = np.random.randint(1, 8)
        ker = get_disk_footprint(ker_size, dim3=False)
        y = white_tophat(x, footprint=ker)

        y = zero_spatial_edges(y)

        if np.random.rand() < 0.3:
            thr = np.random.randint(1, 30)
            y[y < thr] = 0
        else:
            area_size = np.random.randint(2, min(20, ker_size * 4))
            area_thr = np.random.randint(5, 25)
            y = remove_small_components(y, area_size, area_thr, 4)

        return y