import os
import numpy as np
import torch
from torch.utils.data import Dataset


class STEMDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir

        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        assert len(self.input_files) == len(self.target_files)

        # build index map
        self.index_map = []
        self.file_shapes = []
        for in_f, tar_f in zip(self.input_files, self.target_files):
            inp = np.load(os.path.join(input_dir, in_f), mmap_mode='r')
            self.file_shapes.append(len(inp))
            for i in range(len(inp)):
                self.index_map.append((in_f, tar_f, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        in_f, tar_f, img_idx = self.index_map[idx]
        x = np.load(os.path.join(self.input_dir, in_f), mmap_mode='r')[img_idx]
        y = np.load(os.path.join(self.target_dir, tar_f), mmap_mode='r')[img_idx]

        if x.ndim == 2:
            x = x[None, ...]
        if y.ndim == 2:
            y = y[None, ...]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y