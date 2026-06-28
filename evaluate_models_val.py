from unet.model import ResidualUNet
from unet import to_onnx
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import h5py
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
    "2D": ResidualUNet.load("unet/runs/20260625_181213_2D_lr0.0001_nc11810_lTrue_HuberLoss/residual_unet_epoch20.pt"),
}

model_dir = "20260628_013819_self_sup_all_lr6e-4_min_lr2e-7_25epochs"

for i in range(5, 26):
    models[f"Self Supervised - Epoch {i}"] = ResidualUNet.load(f"unet/runs/{model_dir}/residual_unet_epoch{i}.pt")

metrics = [
    reverse_kl_divergence,
    symmetric_cross_entropy,
    symmetric_mean_absolute_percentage_error
]

db_dir = Path("unet/dataset/dbase/")
results_dir = Path("evaluation_results_val") / model_dir
models_dir = results_dir / "models"
models_dir.mkdir(exist_ok=True, parents=True)

all_results = []
for name, (m, p) in tqdm(models.items()):
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

# 3. Calculate the mean over the samples
# We group by 'Method/Model' and 'Deconv' so that the 'Sample' column is averaged out
df_mean = df.groupby(["Method/Model", "Deconv"]).mean(numeric_only=True).reset_index()

# 4. Save the DataFrames to CSV files
df.to_csv(results_dir / "detailed_scores.csv", index=False)
df_mean.to_csv(results_dir / "mean_scores.csv", index=False)

# Optional: Display the mean scores in the console
print("--- Mean Scores Over Samples ---")
print(df_mean.set_index(["Method/Model", "Deconv"]))
