from examples.sum.sum_fn import load_cached
from pathlib import Path
import numpy as np
import stemdiff
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("."), './unet/')))
from dataset_enhancement import save_h5

def split_df(df, use_max_rows=8_000):
    # 1. Shuffle the entire dataframe
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled = df_shuffled.iloc[:use_max_rows]

    # 2. Define split sizes
    train_size = int(0.7 * len(df_shuffled))
    val_size = int(0.15 * len(df_shuffled))

    # 3. Slice the dataframe
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size : train_size + val_size]
    test_df = df_shuffled.iloc[train_size + val_size :]

    return train_df, val_df, test_df

def split_subfolders(df: pd.DataFrame, groups: dict) -> list[pd.DataFrame]:
    grouped = df.groupby(lambda idx: str(df.loc[idx, 'DatafileName'].parent))
    return pd.concat([grouped.get_group(key) for key in groups["train"]]), \
           pd.concat([grouped.get_group(key) for key in groups["val"]]), \
           pd.concat([grouped.get_group(key) for key in groups["test"]]) if "test" in groups else None

def load_arrays(SDATA, df):
    arrs = []
    for index, datafile in df.iterrows():
        datafile_name = SDATA.data_dir.joinpath(datafile.DatafileName)
        arr = stemdiff.io.Datafiles.read(SDATA, datafile_name)
        arrs.append(arr)

    return np.stack(arrs)

# Commented out to use all as a final test set
paths = [
    "DATA.STEMDIFF/1_AU/EX1.AU/DATA",
    "DATA.STEMDIFF/2_TBF3/VZ2.TBF3.R2",
    "DATA.STEMDIFF/3_FEO_PURE/FeO-Pure_Cimc",
    "DATA.STEMDIFF/4_MARUSKA_LAF3/D_MARUSKA_C214",
    "DATA.STEMDIFF/X1_GDF3/VZ2.GDF3.R2",
    # "DATA.STEMDIFF/X2_TIO2/VZ4.TIO2-A.M2.R2",
    # "DATA.STEMDIFF/X2_TIO2/VZ4.TIO2-R.M2.R2"
]

names = [
    "au",
    "tbf3",
    "feo",
    "laf3",
    "gdf3",
    # "tio2-a",
    # "tio2-r"
]

paths = list(map(Path, paths))
output_dir = Path("unet/dataset1.1/")

train = {}
val = {}
test = {}
for p, n in zip(paths, names):
    SDATA, DIFFIMAGES, df = load_cached(p, n)

    # Split laf3 and feo by subfolders.
    if n == "laf3":
        df_train, df_val, df_test = split_subfolders(df, {
            "train": ["02", "03"], # 2 * 1600 files
            "val": ["01"], # 400 files
            "test": ["04", "05"] # 2 * 1600 files, higher max intensities
        })
    elif n == "feo":
        df_train, df_val, df_test = split_subfolders(df, {
            "train": ["02", "03"], # 2 * 2400 files
            "val": ["01"], # 2400 files
        })
        df_test = df_val[len(df_val) // 2:]
        df_val = df_val[:len(df_val) // 2]
    else:
        df_train, df_val, df_test = split_df(df)

    train[n] = load_arrays(SDATA, df_train)
    val[n] = load_arrays(SDATA, df_val)
    test[n] = load_arrays(SDATA, df_test)

    print(n, "train:", train[n].shape)
    print(n, "val:", val[n].shape)
    print(n, "test:", test[n].shape)
    print()

    stemdiff.dbase.save_database(df_train, output_dir / f"db_train_{n}")
    stemdiff.dbase.save_database(df_val, output_dir / f"db_val_{n}")
    stemdiff.dbase.save_database(df_test, output_dir / f"db_test_{n}")

save_h5(train, output_dir / "train.h5")
save_h5(val, output_dir / "val.h5")
save_h5(test, output_dir / "test.h5")