import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
import random
from pathlib import Path
import h5py
from dataset_enhancement import *
from skimage.morphology import disk
import pandas as pd
import json


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
        y = self.augment(x, in_key)

        if x.ndim == 2:
            x = x[None, ...]
        if y.ndim == 2:
            y = y[None, ...]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y
    
    def augment(self, x, name):
        ker_size = np.random.randint(1, 5)
        ker = disk(ker_size)
        y = cv.morphologyEx(x, cv.MORPH_TOPHAT, ker)

        y = zero_spatial_edges(y)

        if np.random.rand() < 0.3:
            if name == "laf3":
                thr = np.random.randint(40, 80)
            else:
                thr = np.random.randint(5, 45)
            y[y < thr] = 0
        else:
            area_size = np.random.randint(2, ker_size * 3)

            if name == "laf3":
                area_thr = np.random.randint(30, 70)
            else:
                area_thr = np.random.randint(5, 25)

            y = remove_small_components(y, area_size, area_thr, 4)

        return y

class ResizedAugmentedDataset(AugmentedDataset):
    def __getitem__(self, idx, scale_factor=4):
        # This part ensures each Worker Process gets its own file handle
        if self.in_fh is None:
            self.in_fh = h5py.File(self.input_h5, 'r')

        in_key, img_idx = self.index_map[idx]

        x = self.in_fh[in_key][img_idx]
        if scale_factor != 1:
            x = torch.from_numpy(x.astype(np.float32))
            # x = skimage.transform.resize(x, (1024, 1024), order=3, anti_aliasing=False)
            x = torch.nn.functional.interpolate(x[None, None], scale_factor=scale_factor, mode="bicubic")
            x = x.squeeze(0)
            y = self.augment(x.squeeze().numpy(), in_key)
        else:
            y = self.augment(x, in_key)


        if len(x.shape) == 2:
            x = x[None, ...]
        if y.ndim == 2:
            y = y[None, ...]

        y = torch.from_numpy(y.astype(np.float32))

        return x, y
    
    def augment(self, x, name):
        sigma = np.random.rand() * 15 + 0.5
        b = cv.GaussianBlur(x, (0, 0), sigma)
        b = np.clip(b, 0, x)
        y = x - b

        y = zero_spatial_edges(y, 10)

        if np.random.rand() < 0.25:
            thr = np.random.randint(5, 25)
            y[y < thr] = 0
        else:
            area_size = np.random.randint(2, int(sigma + 1) * 3)
            area_size = min(area_size, 30)

            area_thr = np.random.randint(5, 25)

            y = remove_small_components(y, area_size, area_thr, 4)

        return y

class Profile1DDataset(Dataset):
    def __init__(self, dataset_path, target_dir):
        self.input_h5 = dataset_path
        target_dir = Path(target_dir)
        self.index_map = []

        f_in = h5py.File(self.input_h5, 'r')
        in_keys = sorted(list(f_in.keys()))

        for in_k in in_keys:
            in_len = f_in[in_k].shape[0]

            for i in range(in_len):
                self.index_map.append((in_k, i))
        
        # We'll open the files lazily in __getitem__
        self.in_fh = None
        
        with open(target_dir / "center_sizes.json", "r") as f:
            center_sizes = json.load(f)

        self.profiles = {}
        for key in sorted(f_in.keys()):
            df = pd.read_csv(target_dir / key , sep=r'\s+')
            self.profiles[key] = (df.q.to_numpy(), df.I.to_numpy(), center_sizes[key])


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # This part ensures each Worker Process gets its own file handle
        if self.in_fh is None:
            self.in_fh = h5py.File(self.input_h5, 'r')

        in_key, img_idx = self.index_map[idx]

        x = self.in_fh[in_key][img_idx]

        if x.ndim == 2:
            x = x[None, ...]

        x = torch.from_numpy(x.astype(np.float32))

        return x, self.profiles[in_key]
    

class SameKeyBatchSampler(Sampler):
    def __init__(self, index_map, batch_size, drop_last=False, shuffle=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 1. Group indices by their in_key
        self.groups = {}
        for idx, (in_key, _) in enumerate(index_map):
            if in_key not in self.groups:
                self.groups[in_key] = []
            self.groups[in_key].append(idx)

    def __iter__(self):
        # 2. Break each group into chunks (batches)
        all_batches = []
        for in_key in self.groups:
            indices = self.groups[in_key][:] # Copy list
            if self.shuffle:
                random.shuffle(indices)
            
            # Group into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # 3. Shuffle the list of batches
        # This ensures you don't train on Key 1 for 100 steps, then Key 2...
        # Instead, batches from different keys are mixed together.
        if self.shuffle:
            random.shuffle(all_batches)
        
        yield from all_batches

    def __len__(self):
        count = 0
        for indices in self.groups.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count