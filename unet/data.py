import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
import random
from pathlib import Path
import h5py
from dataset_enhancement import *
import json
import stemdiff as sd

def rescale(img, scale_factor=0, dsize=None):
    resized = cv.resize(
        img.astype(np.float32),
        dsize, 
        fx=scale_factor, 
        fy=scale_factor, 
        interpolation=cv.INTER_CUBIC
    )
    resized_max = resized.max()
    if resized_max > 0.01:
        resized = resized / resized_max * img.max()
    return np.clip(resized, 0, None)

class STEMDataset(Dataset):
    def __init__(self, dataset_dir, input_fname, target_fname, scale_factor=1,
                 include_profiles=False, profile_scale=1):
        dataset_dir = Path(dataset_dir)
        self.input_h5 = dataset_dir / input_fname
        self.target_h5 = dataset_dir / target_fname if target_fname is not None else None
        self.scale_factor = scale_factor
        self.index_map = dataset_dir
        self.include_profiles = include_profiles

        f_in = h5py.File(self.input_h5, 'r')
        in_keys = sorted(list(f_in.keys()))

        if self.target_h5 is None:
            for in_k in in_keys:
                in_len = f_in[in_k].shape[0]

                for i in range(in_len):
                    self.index_map.append((in_k, i))
        else:
            f_tar = h5py.File(self.target_h5, 'r')
            tar_keys = sorted(list(f_tar.keys()))

            for in_k, tar_k in zip(in_keys, tar_keys):
                in_len = f_in[in_k].shape[0]
                tar_len = f_tar[tar_k].shape[0]

                if in_len != tar_len:
                    raise ValueError(f"Length mismatch at {in_k} ({in_len}) and {tar_k} ({tar_len})")

                for i in range(in_len):
                    self.index_map.append((in_k, tar_k, i))

            # We'll open the files lazily in __getitem__
            self.tar_fh = None
        self.in_fh = None

        if not include_profiles:
            return
        
        with open(dataset_dir / "dataset_params.json", "r") as f:
            dataset_params = json.load(f)
            self.center_sizes = dataset_params["center_sizes"]
            cal_consts = dataset_params["calibration_constants"]

        self.profiles = {}
        self.centers = {}
        for key in sorted(f_in.keys()):
            profile_fname = key + (f"x{profile_scale}" if profile_scale != 1 else "")
            p = np.loadtxt(dataset_dir / "target_profiles" / profile_fname)
            self.profiles[key] = p

            db_path = dataset_dir / "dbase" / f"db_{self.input_h5.stem}_{key}"
            db = sd.dbase.read_database(db_path)
            c = db[["Xcenter", "Ycenter"]].to_numpy()
            c /= 4 # Centers are for 4x images
            self.centers[key] = c


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # This part ensures each Worker Process gets its own file handle
        if self.in_fh is None:
            self.in_fh = h5py.File(self.input_h5, 'r')
        if self.target_h5 is not None and self.tar_fh is None:
            self.tar_fh = h5py.File(self.target_h5, 'r')

        # Load data from file
        if self.target_h5 is None:
            in_key, img_idx = self.index_map[idx]
        else:
            in_key, tar_key, img_idx = self.index_map[idx]
            y = self.tar_fh[tar_key][img_idx]
        x = self.in_fh[in_key][img_idx]

        # Rescale, convert and return
        if self.scale_factor != 1:
            x = rescale(x, self.scale_factor)

        if x.ndim == 2:
            x = x[None, ...]
        x = torch.from_numpy(x.astype(np.float32))

        if self.target_h5 is None:
            return x, (
                        self.profiles[in_key], 
                        self.center_sizes[in_key],
                        self.centers[in_key][img_idx]
                    )
        else:
            if x.shape != y.shape:
                y = rescale(y, dsize=x.shape[1:])
            if y.ndim == 2:
                y = y[None, ...]
            y = torch.from_numpy(y.astype(np.float32))

            if self.include_profiles:
                return x, (
                            y, 
                            self.profiles[in_key], 
                            self.center_sizes[in_key],
                            self.centers[in_key][img_idx]
                        )
            else:
                return x, y
    

class SameKeyBatchSampler(Sampler):
    def __init__(self, index_map, batch_size, drop_last=False, shuffle=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 1. Group indices by their in_key
        self.groups = {}
        for idx, val in enumerate(index_map):
            in_key = val[0]
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