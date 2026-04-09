import numpy as np
from plot import show_1D_profiles, show_diffractograms
import torch
from pathlib import Path

def predict(model, data):
    data = data.astype(np.float32)
    data = torch.from_numpy(data)

    output = model.batch_predict(data)

    return output.squeeze().numpy()

def zero_spatial_edges(data, border_width=1):
    """
    Zeros the edges of an array with shape (..., H, W).
    Works for (C, H, W) and (B, C, H, W).
    """
    res = data.copy()
    w = border_width
    
    # Zero Top and Bottom
    res[..., :w, :] = 0      # All batches/channels, first 'w' rows
    res[..., -w:, :] = 0     # All batches/channels, last 'w' rows
    
    # Zero Left and Right
    res[..., :, :w] = 0      # All batches/channels, first 'w' columns
    res[..., :, -w:] = 0     # All batches/channels, last 'w' columns
    
    return res

def process(name, model, thr, show_n=5, clip_max=1):
    # Load
    data = np.load(f"dataset/x_input/input{name}.npy")
    target = np.load(f"dataset/x_target/target{name}.npy")

    # Enhance with unet
    output = predict(model, data)

    # Enhance output
    noise_rem = output.copy()
    noise_rem[noise_rem<thr]=0
    noise_rem = zero_spatial_edges(noise_rem)

    for i in range(show_n):
        show_diffractograms({
                "Original": data[i], 
                "Result": output[i],
                "Noise removal": noise_rem[i],
                "Target": target[i]
            }, clip_max)
        show_1D_profiles({
                "Result": (output[i], "green"), 
                "Noise removal": (noise_rem[i], "-.m")
            })

    return noise_rem.astype(np.uint16)

def save_results(results, output_dir):
    output_dir = Path(output_dir)
    new_input = output_dir / "x_input.npz"
    new_target = output_dir / "x_target.npz"
    orig_data = Path(f"dataset/x_input.npz")

    output_dir.mkdir(parents=True, exist_ok=True)


    if not new_input.exists():
        new_input.symlink_to(".." / orig_data)

    np.savez_compressed(new_target, **results)