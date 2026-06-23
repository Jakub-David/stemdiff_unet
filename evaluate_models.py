from unet.model import ResidualUNet
from unet import to_onnx
from examples.sum.sum_fn import load_cached, filter_datafiles, create_profile
from pathlib import Path
from typing import Callable
import stemdiff as sd
import ediff as ed
import pandas as pd
import numpy as np
import h5py
from grid_search import (
    kl_divergence, 
    reverse_kl_divergence, 
    symmetric_cross_entropy, 
    symmetric_mean_absolute_percentage_error,

)

def evaluate_sample(sample_name: str, metrics: list[Callable], bkg: int, bkgp: dict, deconv: int, visualize=False):
    SDATA, DIFFIMAGES, df_all = load_cached(
        Path(samples[sample_name]), 
        sample_name,
        "unet/dataset/dbase",
        f"db_test_{sample_name}"
    )

    df = filter_datafiles(df_all, 100)

    XRD = ed.pcryst.XRD_polycrystal(
        structure=cif_paths[sample_name],
        wavelength=0.71,
        two_theta_range=(5, 100),
        peak_profile_sigma=0.03,
    )
    # XRD.diffractogram = pd.read_csv(f"unet/dataset/{sample_name}", sep=r'\s+')

    summed = sd.summ.sum_datafiles(
        SDATA, 
        DIFFIMAGES, 
        df, 
        bkg=bkg, 
        bkgp=bkgp,
        deconv=deconv
    )
    ELD = create_profile(
        summed,
        XRD,
        (30, 800),
        xrd_range=None,
        show=visualize,
        center=None,
        in_file="examples/sum/center.txt",
    )

    eld_diff = ELD.diffractogram
    xrd_diff = XRD.diffractogram

    scores = {}
    for metric in metrics:
        if eld_diff.I.sum() < 1.2:
            score = np.inf
        else:
            eld_I = np.interp(xrd_diff.q, eld_diff.q, eld_diff.I)
            score = metric(xrd_diff.I, eld_I)
        
        scores[metric.__name__] = score

    return scores

samples = {
    "au": "DATA.STEMDIFF/1_AU/EX1.AU/DATA",
    "tbf3": "DATA.STEMDIFF/2_TBF3/VZ2.TBF3.R2",
    "feo": "DATA.STEMDIFF/3_FEO_PURE/FeO-Pure_Cimc",
    "laf3": "DATA.STEMDIFF/4_MARUSKA_LAF3/D_MARUSKA_C214",
    "gdf3": "DATA.STEMDIFF/X1_GDF3/VZ2.GDF3.R2",
    "tio2-a": "DATA.STEMDIFF/X2_TIO2/VZ4.TIO2-A.M2.R2",
    "tio2-r": "DATA.STEMDIFF/X2_TIO2/VZ4.TIO2-R.M2.R2"
}

cif_paths = {
    "au": "DATA.STEMDIFF/cif/au_9008463.cif",
    "tbf3": "DATA.STEMDIFF/cif/1530594_tbf3.cif",
    "feo": "DATA.STEMDIFF/cif/Fe3O4.cif",
    "laf3": "DATA.STEMDIFF/cif/laf3_9008114.cif",
    "gdf3": "DATA.STEMDIFF/cif/1530594_gdf3.cif",
    "tio2-a": "DATA.STEMDIFF/cif/tio2_anatase_9015929.cif",
    "tio2-r": "DATA.STEMDIFF/cif/tio2_rutile_9015662.cif",
}

# xranges = {
#     "au": (30, 800),
#     "tbf3": (30, 800),
#     "feo": (30, 800),
#     "laf3": (30, 800),
#     "gdf3": (30, 800),
#     "tio2-a": (30, 800),
#     "tio2-r": (30, 800),
# }

models = {
    "2D": ResidualUNet.load("unet/runs/20260621_183607_2D_lr0.0001_nc11810_lTrue_HuberLoss/residual_unet_epoch20.pt"),
    "self sup": ResidualUNet.load("unet/runs/20260622_132820_self_sup_ncNone_lcw0.01_l1w0.001/residual_unet_epoch20.pt")
}

metrics = [
    kl_divergence,
    reverse_kl_divergence,
    symmetric_cross_entropy,
    symmetric_mean_absolute_percentage_error,
]

db_dir = Path("unet/dataset/dbase/")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    all_results = []
    for name, (m, p) in models.items():
        print("Evaluating model:", name)
        model_onnx_path = models_dir / f"{name}.onnx"
        to_onnx.convert(
            m, 
            model_onnx_path, 
            h5py.File("unet/dataset/train.h5", 'r'),
            verbose=False
        )

        for sample_name in samples:
            print("Processing sample:", sample_name)
            for deconv in [0, 1]:
                scores = evaluate_sample(
                    sample_name, 
                    metrics,
                    4,
                    {"path": model_onnx_path},
                    deconv,
                    visualize=False
                )

                # Capture the metadata and unpack the scores dict
                row = {
                    "Method/Model": f"ONNX ({name})",
                    "Sample": sample_name,
                    "Deconv": deconv,
                    **scores
                }
                all_results.append(row)

    print("Evaluating Gaussian")
    for sample_name in samples:
        print("Processing sample:", sample_name)
        for deconv in [0, 1]:
            scores = evaluate_sample(
                sample_name, 
                metrics,
                3,
                {
                    "sigma": 2.2,
                    "thr": 2.2,
                    "area_size": 7.2,
                    "normalize": True
                },
                deconv,
                visualize=False
            )

            # Capture metadata and unpack scores gaussian
            row = {
                "Method/Model": "Baseline (Gaussian)",
                "Sample": sample_name,
                "Deconv": deconv,
                **scores
            }
            all_results.append(row)

    
    # 2. Convert to a DataFrame
    df = pd.DataFrame(all_results)

    # 3. Calculate the mean over the samples
    # We group by 'Method/Model' and 'Deconv' so that the 'Sample' column is averaged out
    df_mean = df.groupby(["Method/Model", "Deconv"]).mean(numeric_only=True).reset_index()

    # 4. Save the DataFrames to CSV files
    df.to_csv("detailed_scores.csv", index=False)
    df_mean.to_csv("mean_scores.csv", index=False)

    # Optional: Display the mean scores in the console
    print("--- Mean Scores Over Samples ---")
    print(df_mean.set_index(["Method/Model", "Deconv"]))
