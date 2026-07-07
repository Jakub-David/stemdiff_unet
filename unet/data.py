import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, Dataset
import random
from pathlib import Path
import h5py
import cv2 as cv
import stemdiff as sd

CALIBRATION_CONSTANTS = {
    'au': 0.03377241772151899, 
    'tbf3': 0.031983907407407405, 
    'feo': 0.031019912500000003, 
    'laf3': 0.03087730158730159, 
    'gdf3': 0.03332153703703704, 
    'tio2-a': 0.03191196428571429, 
    'tio2-r': 0.03171350819672131, 
    'feo_shell': 0.031111495408000765 * 0.99
}


def resize_profile(q, I, calibration_constant, profile_scale) -> torch.Tensor:
    # Calibration constant is ELD -> XRD for images resized 4x
    calibration_constant = 1 / calibration_constant # XRD -> ELD
    calibration_constant = calibration_constant / 4 # for profile_scale 1
    calibration_constant = calibration_constant * profile_scale

    # 0. Define the number of bins
    N = torch.round(q[-1] * calibration_constant).int()

    # 1. Round x and convert to long/int so it can be used as indices
    # We also clip the values to ensure they stay within [0, N-1]
    indices = torch.round(q * calibration_constant).long().clamp(0, N - 1)

    # 2. Initialize the output tensor of length N
    out = torch.zeros((N,), dtype=I.dtype, device=I.device)

    # 3. Use scatter_reduce to find the maximum for each index
    out.scatter_reduce_(dim=0, index=indices, src=I, reduce="amax", include_self=False)

    return out

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
                 include_profiles=False, profile_scale=1, 
                 profiles_dir=Path("../DATA.STEMDIFF/profiles")):
        dataset_dir = Path(dataset_dir)
        self.input_h5 = dataset_dir / input_fname
        self.target_h5 = dataset_dir / target_fname if target_fname is not None else None
        self.scale_factor = scale_factor
        self.index_map = []
        self.include_profiles = include_profiles

        f_in = h5py.File(self.input_h5, 'r')
        in_keys = sorted(list(f_in.keys()))

        if self.target_h5 is None:
            for in_k in in_keys:
                in_len, h, w = f_in[in_k].shape

                for i in range(in_len):
                    self.index_map.append((in_k, i))
        else:
            f_tar = h5py.File(self.target_h5, 'r')
            tar_keys = sorted(list(f_tar.keys()))

            for in_k, tar_k in zip(in_keys, tar_keys):
                in_len, h, w = f_in[in_k].shape
                tar_len = f_tar[tar_k].shape[0]

                if in_len != tar_len:
                    raise ValueError(f"Length mismatch at {in_k} ({in_len}) and {tar_k} ({tar_len})")

                for i in range(in_len):
                    self.index_map.append((in_k, tar_k, i))

            # We'll open the files lazily in __getitem__
            self.tar_fh = None
        self.in_fh = None

        # Define border mask for noise estimation
        cx, cy = h // 2, w // 2
        r = round(h / 2.5)  # Safe radius to avoid the central peak

        bg_mask = np.ones((1, h, w), dtype=bool)
        bg_mask[0, cx-r:cx+r, cy-r:cy+r] = False
        # Zero out the top 2 and bottom 2 rows
        bg_mask[..., :2, :] = False
        bg_mask[..., -2:, :] = False

        # Zero out the left 2 and right 2 columns
        bg_mask[..., :, :2] = False
        bg_mask[..., :, -2:] = False

        self.bg_mask = bg_mask
        self.std_devs = {}

        if not include_profiles:
            return

        self.profiles = {}
        self.centers = {}
        max_prof_len = 0
        for key in sorted(f_in.keys()):
            df = pd.read_csv(profiles_dir / key, sep=r'\s+')
            q = torch.from_numpy(df.q.to_numpy()).float()
            I = torch.from_numpy(df.I.to_numpy()).float()
            p = resize_profile(q, I, CALIBRATION_CONSTANTS[key], profile_scale)
            self.profiles[key] = p
            max_prof_len = max(max_prof_len, len(p))

            db_path = dataset_dir / "dbase" / f"db_{self.input_h5.stem}_{key}"
            db = sd.dbase.read_database(db_path)
            c = db[["Xcenter", "Ycenter"]].to_numpy().astype(np.float32)
            c /= 4 # Centers are for 4x images
            self.centers[key] = c

        # Pad profiles to allow batching
        for key in sorted(f_in.keys()):
            pad_width = max_prof_len - len(self.profiles[key])
            self.profiles[key] = np.pad(self.profiles[key], (0, pad_width))


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

        # Estimate noise
        std = self.std_devs.get(idx)
        if std is None:
            std = x[self.bg_mask].std().reshape((1, 1, 1))
            self.std_devs[idx] = std

        result = {
            "original_image": x,
            "noise_std": std
        }

        if self.target_h5 is not None:
            if x.shape != y.shape:
                y = rescale(y, dsize=x.shape[1:])
            if y.ndim == 2:
                y = y[None, ...]
            y = torch.from_numpy(y.astype(np.float32))

            result["target_2d"] = y

        if self.include_profiles:
            result["target_profile"] = self.profiles[in_key]
            result["center_size"] = 6
            result["center"] = self.centers[in_key][img_idx]
            
        return result
    

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