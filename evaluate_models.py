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
import matplotlib.pyplot as plt
import pickle
from grid_search import (
    reverse_kl_divergence, 
    symmetric_cross_entropy, 
    symmetric_mean_absolute_percentage_error,

)

def evaluate_sample(sample_name: str, metrics: list[Callable], bkg: int, bkgp: dict, deconv: int, visualize=False):
    if bkg == 3:
        run_name = "gaussian"
    else:
        run_name =  bkgp["path"].stem
    current_results_dir = results_dir / run_name
    current_results_dir.mkdir(exist_ok=True)


    SDATA, DIFFIMAGES, df_all = load_cached(
        Path(samples[sample_name]), 
        sample_name,
        "unet/dataset/dbase",
        # TODO: change back to test
        f"db_val_{sample_name}"
    )

    df = filter_datafiles(df_all, 100)

    XRD = ed.pcryst.XRD_polycrystal(
        structure=cif_paths[sample_name],
        wavelength=0.71,
        two_theta_range=(5, 100),
        peak_profile_sigma=0.03,
    )
    # xrd_diff_path = Path(f"unet/dataset/{sample_name}")
    # if xrd_diff_path.exists():
    #     XRD.diffractogram = pd.read_csv(xrd_diff_path, sep=r'\s+')

    # create psf
    if deconv == 1:
        df_psf = df_all[(df_all.Peaks > 0)]
        df_psf = df_psf.sort_values(by=['Peaks', 'S'], ascending=[True, True])
        df_psf = df_psf[:100]
        psf = sd.summ.sum_datafiles(
            SDATA, 
            DIFFIMAGES, 
            df,
            bkg=bkg, 
            bkgp=bkgp,
        )
        c = psf.shape[0] // 2
        cs = 20
        psf = psf[c-cs:c+cs, c-cs:c+cs]
        psf[psf < 100] = 0

        plt.figure()
        plt.imshow(np.log1p(psf), vmin=0, vmax=1)
        plt.savefig(current_results_dir / f"{sample_name}_psf.png")
        np.savetxt(current_results_dir / "psf", psf)
        if visualize:
            plt.show()
    else:
        psf = None


    # sum datafiles
    summed = sd.summ.sum_datafiles(
        SDATA, 
        DIFFIMAGES, 
        df, 
        bkg=bkg, 
        bkgp=bkgp,
        deconv=deconv,
        deconvp={"num_iter": 10, "psf": psf}
    )

    if sample_name in ["tbf3", "gdf3"]:
        xrd_range = (0, 1.9)
    elif sample_name == "feo_shell":
        xrd_range = (0, 2.2)
    else:
        xrd_range = None
    ELD = create_profile(
        summed,
        XRD,
        (30, 800),
        xrd_range=xrd_range,
        show=False,
        center=None,
        in_file="examples/sum/center.txt",
    )

    if calibration_constants is not None:
        ELD.diffractogram.q = ELD.diffractogram.Pixels * calibration_constants[sample_name]

    with open(current_results_dir / f"{sample_name}_eld_deconv{deconv}", "wb") as f:
            pickle.dump(ELD, f)
    plt.figure()
    plt.title(run_name)
    ELD.compare_with_XRD(
        XRD, 
        fine_tune=1, 
        Xlim=(0.5, 14), 
        CLI=not visualize,
        out_file=current_results_dir / f"{sample_name}_deconv{deconv}.svg"
    )

    eld_diff = ELD.diffractogram
    xrd_diff = XRD.diffractogram
    eld_I = np.interp(xrd_diff.q, eld_diff.q, eld_diff.I)

    plt.figure()
    plt.plot(xrd_diff.q, xrd_diff.I, label="XRD")
    plt.plot(xrd_diff.q, eld_I, label="ELD")
    plt.legend()
    plt.savefig(current_results_dir / f"{sample_name}_deconv{deconv}_interpolated.svg")

    # if deconv == 1:
    #     calibration_constants[sample_name] = ELD.calibration_constant

    scores = {}
    for metric in metrics:
        score = metric(xrd_diff.I, eld_I)
        scores[metric.__name__] = score

    return scores

samples = {
    "au": "DATA.STEMDIFF/1_AU/EX1.AU/DATA",
    "tbf3": "DATA.STEMDIFF/2_TBF3/VZ2.TBF3.R2",
    "feo": "DATA.STEMDIFF/3_FEO_PURE/FeO-Pure_Cimc",
    "feo_shell": "DATA.STEMDIFF/FeO-Shell_Cimc",
    "laf3": "DATA.STEMDIFF/4_MARUSKA_LAF3/D_MARUSKA_C214",
    "gdf3": "DATA.STEMDIFF/X1_GDF3/VZ2.GDF3.R2",
    "tio2-a": "DATA.STEMDIFF/X2_TIO2/VZ4.TIO2-A.M2.R2",
    "tio2-r": "DATA.STEMDIFF/X2_TIO2/VZ4.TIO2-R.M2.R2"
}

cif_paths = {
    "au": "DATA.STEMDIFF/cif/au_9008463.cif",
    "tbf3": "DATA.STEMDIFF/cif/1530594_tbf3.cif",
    "feo": "DATA.STEMDIFF/cif/Fe3O4.cif",
    "feo_shell": "DATA.STEMDIFF/cif/Fe3O4.cif",
    "laf3": "DATA.STEMDIFF/cif/laf3_9008114.cif",
    "gdf3": "DATA.STEMDIFF/cif/1530594_gdf3.cif",
    "tio2-a": "DATA.STEMDIFF/cif/tio2_anatase_9015929.cif",
    "tio2-r": "DATA.STEMDIFF/cif/tio2_rutile_9015662.cif",
}

calibration_constants = {
    'au': 0.03377241772151899, 
    'tbf3': 0.031983907407407405, 
    'feo': 0.031019912500000003, 
    'laf3': 0.03087730158730159, 
    'gdf3': 0.03332153703703704, 
    'tio2-a': 0.03191196428571429, 
    'tio2-r': 0.03171350819672131, 
    'feo_shell': 0.031111495408000765 * 0.99
}

models = {
    "Self Supervised": ResidualUNet.load("unet/runs/20260625_171745_self_sup_lr0.0008_lc0.55_tv0_bc2_sparse_error_border_std0.5/residual_unet_epoch20.pt"),
    "2D": ResidualUNet.load("unet/runs/20260625_181213_2D_lr0.0001_nc11810_lTrue_HuberLoss/residual_unet_epoch20.pt"),
    # "Self Supervised Wrong": ResidualUNet.load("unet/runs_old/runs4/20260622_132820_wrong_self_sup_ncNone_lcw0.01_l1w0.001/residual_unet_epoch20.pt"),
}

metrics = [
    reverse_kl_divergence,
    symmetric_cross_entropy,
    symmetric_mean_absolute_percentage_error,
]

db_dir = Path("unet/dataset/dbase/")
results_dir = Path("evaluation_results")
results_dir.mkdir(exist_ok=True)
models_dir = results_dir / "models"
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
                    "Method/Model": f"NN ({name})",
                    "Sample": sample_name,
                    "Deconv": deconv,
                    **scores
                }
                all_results.append(row)
        
        # print(calibration_constants)

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
    df.to_csv(results_dir / "detailed_scores.csv", index=False)
    df_mean.to_csv(results_dir / "mean_scores.csv", index=False)

    # Optional: Display the mean scores in the console
    print("--- Mean Scores Over Samples ---")
    print(df_mean.set_index(["Method/Model", "Deconv"]))
