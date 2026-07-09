from unet.model import ResidualUNet
from unet import to_onnx
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import h5py
import scipy.stats
from evaluate_models import evaluate_sample
from grid_search import (
    reverse_kl_divergence, 
    symmetric_cross_entropy, 
    symmetric_mean_absolute_percentage_error
)

samples = {
    "au": "DATA.STEMDIFF/1_AU/EX1.AU/DATA",
    "tbf3": "DATA.STEMDIFF/2_TBF3/VZ2.TBF3.R2",
    "feo": "DATA.STEMDIFF/3_FEO_PURE/FeO-Pure_Cimc",
    "laf3": "DATA.STEMDIFF/4_MARUSKA_LAF3/D_MARUSKA_C214",
    "gdf3": "DATA.STEMDIFF/X1_GDF3/VZ2.GDF3.R2",
}

cif_paths = {
    "au": "DATA.STEMDIFF/cif/au_9008463.cif",
    "tbf3": "DATA.STEMDIFF/cif/1530594_tbf3.cif",
    "feo": "DATA.STEMDIFF/cif/Fe3O4.cif",
    "laf3": "DATA.STEMDIFF/cif/laf3_9008114.cif",
    "gdf3": "DATA.STEMDIFF/cif/1530594_gdf3.cif",
}

calibration_constants = {
    'au': 0.03377241772151899, 
    'tbf3': 0.031983907407407405, 
    'feo': 0.031019912500000003, 
    'laf3': 0.03087730158730159, 
    'gdf3': 0.03332153703703704, 
}

models = {
    "2D": ResidualUNet.load("unet/runs/20260708_184202_2D_bc4_lr0.001_lTrue_nc11810/residual_unet_epoch14.pt"),
    "Self Supervised": ResidualUNet.load("unet/runs/20260709_001306_self_sup_lr0.001_min_lr0.0001_lc0.55_c0_bc4_lFalse/residual_unet_epoch4.pt"),
}

runs_dir = Path("unet/runs")

# model_paths = runs_dir.glob("*2D_bc4*/*.pt")
# training_type = "2D"
# for p in model_paths:
#     time = p.parent.name.split("_")[1]
#     epoch = p.stem.removeprefix("residual_unet_epoch")
#     # if time < "163929":
#     #     continue
#     if "lr0.0001" in p.parent.name and "lFalse_ncNone" not in p.parent.name:
#         continue
#     models[f"{training_type} - {time} e{epoch}"] = ResidualUNet.load(p)

# model_paths = runs_dir.glob("*self_sup_lr*/*.pt")
# training_type = "Self Supervised"
# for p in model_paths:
#     time = p.parent.name.split("_")[1]
#     epoch = p.stem.removeprefix("residual_unet_epoch")
#     models[f"{training_type} - {time} e{epoch}"] = ResidualUNet.load(p)


model_paths = runs_dir.glob("*self_sup_all*/*.pt")
training_type = "Self Supervised All Data"
for p in model_paths:
    time = p.parent.name.split("_")[1]
    epoch = p.stem.removeprefix("residual_unet_epoch")
    models[f"{training_type} - {time} e{epoch}"] = ResidualUNet.load(p)

metrics = [
    reverse_kl_divergence,
    symmetric_cross_entropy,
    symmetric_mean_absolute_percentage_error
]

db_dir = Path("unet/dataset/dbase/")
results_dir = Path("evaluation_results_val") / training_type
models_dir = results_dir / "models"
models_dir.mkdir(exist_ok=True, parents=True)

all_results = []
for name, (m, p) in tqdm(models.items()):
    tqdm.write("Evaluating model: " + name)
    model_onnx_path = models_dir / f"{name}.onnx"
    to_onnx.convert(
        m, 
        model_onnx_path, 
        h5py.File("unet/dataset/train.h5", 'r'),
        verbose=False
    )

    for sample_name in samples:
        print("Processing sample:", sample_name)
        for deconv in [0]:
            scores = evaluate_sample(
                sample_name, 
                metrics,
                4,
                {"path": model_onnx_path},
                deconv,
                results_dir,
                samples,
                cif_paths,
                calibration_constants,
                visualize=False,
                split_name="val"
            )

            # Capture the metadata and unpack the scores dict
            row = {
                "Method/Model": f"NN ({name})",
                "Sample": sample_name,
                "Deconv": deconv,
                **scores
            }
            all_results.append(row)


# 2. Convert to a DataFrame
df = pd.DataFrame(all_results)

# 3. Add geo mean
metric_cols = [metric.__name__ for metric in metrics]
df["error geo mean"] = scipy.stats.gmean(df[metric_cols], axis=1)

# 4. Calculate the mean over the samples
# We group by 'Method/Model' and 'Deconv' so that the 'Sample' column is averaged out
df_mean = df.groupby(["Method/Model", "Deconv"]).mean(numeric_only=True).reset_index()

# 5. Save the DataFrames to CSV files
df.to_csv(results_dir / "detailed_scores.csv", index=False)
df_mean.to_csv(results_dir / "mean_scores.csv", index=False)

# Optional: Display the mean scores in the console
print("--- Mean Scores Over Samples ---")
print(df_mean.set_index(["Method/Model", "Deconv"]))
