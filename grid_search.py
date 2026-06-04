import itertools
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error

from examples.sum.sum_fn import *
import ediff as ed

# ==========================================
# 1. CONFIGURATION & DATASETS
# ==========================================
DATA_DIR = Path("DATA.STEMDIFF/")

DATASETS = {
    "au": {
        "path": DATA_DIR / "1_AU/EX1.AU/DATA",
        "XRD_PATH": "DATA.STEMDIFF/cif/au_9008463.cif",
        "xrange": (55, 800),
        "xrd_range": None,
        "db_path": "unet/dataset1.1/dbase/",
        "db_file": "db_train_au",
        "filter_count": 100,
        "wavelength": 0.71,
    },
}

# Define the grid search space for bkgp
PARAM_GRID = {
    "sigma": [10, 14, 18],
    "thr": [2.0, 2.5, 3.0],
    "area_size": [3, 5, 7],
}

CACHE_DIR = Path("./grid_search_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
def get_param_combinations(grid):
    """Generates a list of dictionaries for all parameter combinations."""
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def run_grid_search(dataset_name, ds_config, param_combinations, visualize_best=True):
    print(f"\n=== Starting Grid Search for Dataset: {dataset_name} ===")

    # Load base dataset data
    SDATA, DIFFIMAGES, df_all = load_cached(
        ds_config["path"],
        dataset_name,
        ds_config["db_path"],
        db_file=ds_config["db_file"],
    )
    df = filter_datafiles(df_all, ds_config["filter_count"])

    # Load reference XRD
    XRD = ed.pcryst.XRD_polycrystal(
        structure=ds_config["XRD_PATH"],
        wavelength=ds_config["wavelength"],
        two_theta_range=(5, 100),
        peak_profile_sigma=0.1,
    )

    best_score = float("inf")
    best_params = None
    best_prof = None
    results = []

    # Dataset specific cache directory
    ds_cache_dir = CACHE_DIR / dataset_name
    ds_cache_dir.mkdir(exist_ok=True)

    for i, bkgp in enumerate(param_combinations):
        print(f"[{i+1}/{len(param_combinations)}] Testing: {bkgp}")

        # Unique string id for caching this specific run
        param_id = f"sigma_{bkgp['sigma']}_thr_{bkgp['thr']}_area_{bkgp['area_size']}"
        sum_cache_path = ds_cache_dir / f"sum_{param_id}.npy"
        prof_cache_path = ds_cache_dir / f"prof_{param_id}.pkl"

        # --- STEP 1: Summed Results (Cache/Compute) ---
        if sum_cache_path.exists():
            sum_gaussian = np.load(sum_cache_path)
        else:
            sum_gaussian = sd.summ.sum_datafiles(
                SDATA, DIFFIMAGES, df, bkg=3, bkgp=bkgp
            )
            np.save(sum_cache_path, sum_gaussian)

        # --- STEP 2: Profile Results (Cache/Compute) ---
        if prof_cache_path.exists():
            with open(prof_cache_path, "rb") as f:
                prof_gaussian = pickle.load(f)
        else:
            prof_gaussian = create_profile(
                sum_gaussian,
                XRD,
                ds_config["xrange"],
                xrd_range=ds_config["xrd_range"],
                show=False,
                center=None,
                in_file="examples/sum/center.txt",
            )
            with open(prof_cache_path, "wb") as f:
                pickle.dump(prof_gaussian, f)

        # --- STEP 3: Metric Evaluation ---
        diff_g = prof_gaussian.diffractogram
        gaussian_I = np.interp(XRD.diffractogram.q, diff_g.q, diff_g.I)
        score = mean_absolute_error(XRD.diffractogram.I, gaussian_I)

        print(f"--> MAE: {score:.5f}")
        results.append({"params": bkgp, "score": score})

        # Track the best performer
        if score < best_score:
            best_score = score
            best_params = bkgp
            best_prof = prof_gaussian

    print(
        f"\nGrid Search Finished! Best Params: {best_params} | Best MAE: {best_score:.5f}"
    )

    # --- STEP 4: Optional Visualization ---
    if visualize_best and best_prof is not None:
        print("Visualizing best result...")
        ed.io.Plots.plot_final_eld_and_xrd(
            best_prof.diffractogram,
            XRD.diffractogram,
            eld_data_label=f"Best Gaussian {best_params}",
            fine_tune=1,
            Xlim=(0, 14),
        )

    return best_params, best_score, results


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    combinations = get_param_combinations(PARAM_GRID)

    # Loop through all datasets configured
    for ds_name, ds_config in DATASETS.items():
        best_p, best_s, all_res = run_grid_search(
            ds_name, ds_config, combinations, visualize_best=True
        )