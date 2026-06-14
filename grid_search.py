import itertools
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.special
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functools import partialmethod
from datetime import datetime

from examples.sum.sum_fn import *
import ediff as ed

# ==========================================
# 0. DEFINE METRICS
# ==========================================

def kl_divergence(x, y):
    # Tiny constant (epsilon)
    epsilon = 1e-12

    # Smooth the arrays by adding epsilon to all elements
    x_smoothed = x + epsilon
    y_smoothed = y + epsilon

    # Normalize
    P = x_smoothed / x_smoothed.sum()
    Q = y_smoothed / y_smoothed.sum()

    return scipy.special.rel_entr(P, Q).sum()

def reverse_kl_divergence(x, y):
    return kl_divergence(y, x)

def symmetric_kl_divergence(x, y):
    return kl_divergence(x, y) + kl_divergence(y, x)

def symmetric_mean_absolute_percentage_error(x, y):
    # Tiny constant (epsilon)
    epsilon = 1e-12

    # Smooth the arrays by adding epsilon to all elements
    x = x + epsilon
    y = y + epsilon

    return (2 / len(x)) * np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))

def cross_entropy(x, y):
    # Tiny constant (epsilon)
    epsilon = 1e-12

    # Smooth only y (for x=0,y=0 we have 0 * np.log(Q))
    y_smoothed = y + epsilon

    # Normalize
    P = x / x.sum()
    Q = y_smoothed / y_smoothed.sum()
    return -np.mean(P * np.log(Q))

def symmetric_cross_entropy(x, y):
    return cross_entropy(x, y) + cross_entropy(y, x)

# ==========================================
# 1. CONFIGURATION & DATASETS
# ==========================================
DATA_DIR = Path("DATA.STEMDIFF/")

DATASETS = {
    "au": {
        "path": DATA_DIR / "1_AU/EX1.AU/DATA",
        "cif_path": "DATA.STEMDIFF/cif/au_9008463.cif",
        "xrd_path": "unet/dataset/au",
        "xrange": (55, 800),
        "xrd_range": None,
        "db_path": "unet/dataset/dbase/",
        "db_file": "db_train_au",
        "filter_count": 100,
    },
    "tbf3": {
        "path": DATA_DIR / "2_TBF3/VZ2.TBF3.R2",
        "cif_path": "DATA.STEMDIFF/cif/1530594_tbf3.cif",
        "xrd_path": "unet/dataset/tbf3",
        "xrange": (30, 800),
        "xrd_range": (0, 1.9),
        "db_path": "unet/dataset/dbase/",
        "db_file": "db_train_tbf3",
        "filter_count": 100,
    },
    "feo": {
        "path": DATA_DIR / "3_FEO_PURE/FeO-Pure_Cimc",
        "cif_path": "DATA.STEMDIFF/cif/Fe3O4.cif",
        "xrd_path": "unet/dataset/feo",
        "xrange": (32, 800),
        "xrd_range": (0, 10),
        "db_path": "unet/dataset/dbase/",
        "db_file": "db_train_feo",
        "filter_count": 100,
    },
    "laf3": {
        "path": DATA_DIR / "4_MARUSKA_LAF3/D_MARUSKA_C214",
        "cif_path": "DATA.STEMDIFF/cif/laf3_9008114.cif",
        "xrd_path": "unet/dataset/laf3",
        "xrange": (32, 800),
        "xrd_range": (0, 10),
        "db_path": "unet/dataset/dbase/",
        "db_file": "db_train_laf3",
        "filter_count": 100,
    },
    "gdf3": {
        "path": DATA_DIR / "X1_GDF3/VZ2.GDF3.R2",
        "cif_path": "DATA.STEMDIFF/cif/1530594_gdf3.cif",
        "xrd_path": "unet/dataset/gdf3",
        "xrange": (30, 800),
        "xrd_range": (0, 1.8),
        "db_path": "unet/dataset/dbase/",
        "db_file": "db_train_gdf3",
        "filter_count": 100,
    },
}

# Define the grid search space for bkgp
PARAM_GRID = {
    "sigma": [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10],
    "thr": [1, 1.5, 2, 2.5, 3, 4, 5, 6],
    "area_size": [3, 4, 5, 6, 7, 8, 10],
    "normalize": [True]
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


def evaluate_single_combination(bkgp, ds_cache_dir, SDATA, DIFFIMAGES, df, XRD, 
                                xrange, xrd_range, metric, recalculate_profiles):
    """
    Worker function executed in parallel for a single parameter combination.
    """
    # Force tqdm to default to disabled inside this process
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    param_id = f"sigma_{bkgp['sigma']}_thr_{bkgp['thr']}_area_{bkgp['area_size']}"
    sum_cache_path = ds_cache_dir / f"sum_{param_id}.npy"
    prof_cache_path = ds_cache_dir / f"prof_{param_id}.pkl"

    # --- STEP 1: Summed Results ---
    if sum_cache_path.exists():
        sum_gaussian = np.load(sum_cache_path)
    else:
        sum_gaussian = sd.sum.sum_datafiles(
            SDATA, DIFFIMAGES, df, bkg=3, bkgp=bkgp
        )
        np.save(sum_cache_path, sum_gaussian)

    # --- STEP 2: Profile Results ---
    if prof_cache_path.exists():
        with open(prof_cache_path, "rb") as f:
            prof_gaussian = pickle.load(f)
        if recalculate_profiles:
            prof_gaussian.subtract_background(None, xrange=xrange)
            prof_gaussian.calibrate_and_normalize('MaxPeaksInRange', XRD, xrd_range=xrd_range, eld_range=None)
    else:
        prof_gaussian = create_profile(
            sum_gaussian,
            XRD,
            xrange,
            xrd_range=xrd_range,
            show=False,
            center=None,
            in_file="examples/sum/center.txt",
        )
        with open(prof_cache_path, "wb") as f:
            pickle.dump(prof_gaussian, f)

    # --- STEP 3: Metric Evaluation ---
    diff_g = prof_gaussian.diffractogram
    # Is sum of profile intensities is small,
    # the background subtraction is too aggressive and images are empty
    if diff_g.I.sum() < 1.2:
        score = np.inf
    else:
        gaussian_I = np.interp(XRD.diffractogram.q, diff_g.q, diff_g.I)
        score = metric(XRD.diffractogram.I, gaussian_I)

    # Return results alongside the profile object (needed for final visualization)
    return {"params": bkgp, "score": score, "prof_gaussian": prof_gaussian}


def run_grid_search_parallel(dataset_name, ds_config, param_combinations, metric, 
                             recalculate_profiles=False, visualize_best=True,
                             verbose=True, result_dir=None, profile_sigma=None):
    print(f"\n=== Starting Parallel Grid Search for Dataset: {dataset_name} ===")

    # Load base dataset data once
    SDATA, DIFFIMAGES, df_all = load_cached(
        ds_config["path"],
        dataset_name,
        ds_config["db_path"],
        db_file=ds_config["db_file"],
    )
    df = filter_datafiles(df_all, ds_config["filter_count"])

    # Load reference XRD once
    XRD = ed.pcryst.XRD_polycrystal(
        structure=ds_config["cif_path"],
        wavelength=0.71,
        two_theta_range=(5, 100),
        peak_profile_sigma=profile_sigma or 0.1,
    )
    
    # Replace XRD diff with altered profile
    if profile_sigma is None:
        XRD.diffractogram = pd.read_csv(ds_config["xrd_path"], sep=r'\s+')

    # Save dataset name to avoid another parameter
    XRD.dataset_name = dataset_name

    ds_cache_dir = CACHE_DIR / dataset_name
    ds_cache_dir.mkdir(exist_ok=True)

    results = []
    best_score = float("inf")
    best_params = None
    best_prof = None

    total_tasks = len(param_combinations)

    # Get rid of generator for ProcessPoolExecutor to work
    # (generator can not be pickled)
    SDATA.filenames = list(SDATA.filenames)
    
    # max_workers=None defaults to the machine's CPU core count
    with ProcessPoolExecutor(max_workers=None) as executor:
        # Submit all tasks to the process pool
        futures = {
            executor.submit(
                evaluate_single_combination,
                bkgp,
                ds_cache_dir,
                SDATA,
                DIFFIMAGES,
                df,
                XRD,
                ds_config["xrange"],
                ds_config["xrd_range"],
                metric,
                recalculate_profiles
            ): bkgp for bkgp in param_combinations
        }

        # Process results as they finish
        with tqdm(total=len(futures)) as pbar:
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    res = future.result()
                    bkgp = res["params"]
                    score = res["score"]
                    prof_gaussian = res["prof_gaussian"]

                    if verbose:
                        tqdm.write(f"[{idx}/{total_tasks}] Finished: {bkgp} --> {metric.__name__}: {score:.5f}")
                    results.append({"params": bkgp, "score": score})

                    # Track the best performer overall
                    if score < best_score:
                        best_score = score
                        best_params = bkgp
                        best_prof = prof_gaussian

                except Exception as exc:
                    failed_params = futures[future]
                    tqdm.write(f"Combination {failed_params} generated an exception: {exc}")
                finally:
                    # Free memory
                    futures.pop(future)
                    pbar.update()

    print("\nGrid Search Finished!")
    print(f"Results for dataset {ds_name} -- Best Params: {best_params} | Best Score: {best_score:.5f}")

    # --- STEP 4: Save result ---
    plot_fname = None
    if result_dir is not None:
        with open(result_dir / f"{dataset_name}_xrd", "wb") as f:
            pickle.dump(XRD, f)
        with open(result_dir / f"{dataset_name}_eld", "wb") as f:
            pickle.dump(best_prof, f)

        plot_fname = result_dir / f"{dataset_name}.svg"

    # --- STEP 5: Optional Visualization ---
    if best_prof is not None:
        plt.figure(figsize=(8, 3))
        ed.io.Plots.plot_multiple_eld_and_xrd(
            best_prof.diffractogram,
            XRD.diffractogram,
            eld_data_label=f"Best Gaussian {best_params}",
            fine_tune=1,
            Xlim=(0, 14),
            CLI=not visualize_best,
            out_file=plot_fname
        )

    return best_params, best_score, results


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    combinations = get_param_combinations(PARAM_GRID)
    metrics = [
        # sklearn.metrics.mean_absolute_error,
        # sklearn.metrics.mean_absolute_percentage_error,
        symmetric_mean_absolute_percentage_error,
        # kl_divergence,
        reverse_kl_divergence,
        # scipy.spatial.distance.jensenshannon,
        # symmetric_kl_divergence,
        # cross_entropy,
        symmetric_cross_entropy
    ]
    
    run_name = f"profile_sigma_0.03"
    for profile_sigma in [None]:
        for metric in metrics:
            result_dir = Path("grid_search_results") / run_name / \
                f"{metric.__name__}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_psigma{profile_sigma}"
            result_dir.mkdir(parents=True)

            for ds_name, ds_config in DATASETS.items():
                best_p, best_s, all_res = run_grid_search_parallel(
                    ds_name, 
                    ds_config, 
                    combinations,
                    metric,
                    recalculate_profiles=True,
                    visualize_best=False,
                    verbose=False,
                    result_dir=result_dir,
                    profile_sigma=profile_sigma
                )