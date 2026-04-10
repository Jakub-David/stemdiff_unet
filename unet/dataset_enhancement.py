import numpy as np
from plot import show_1D_profiles, show_diffractograms
import torch
from pathlib import Path
import h5py
import cv2 as cv

def predict(model, data):
    data = data.astype(np.float32)
    data = torch.from_numpy(data)

    output = model.batch_predict(data)

    return output.squeeze().numpy()

def zero_spatial_edges(data, border_width=3):
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

def process(name, model, thr=None, area_size=None, area_thr=1, 
            show_n=5, clip_max=1):
    # Load
    with h5py.File("dataset/x_input.h5", 'r') as f_in, \
             h5py.File("dataset/x_target.h5", 'r') as f_tar:
        data = f_in[f"input{name}"][:]
        target = f_tar[f"target{name}"][:]

    # Enhance with unet
    output = predict(model, data)

    diff_dict = {
                "Original": data, 
                "Result": output
    }

    # Enhance output
    noise_rem = output.copy()
    if thr != None:
        noise_rem[noise_rem<thr]=0
        noise_rem = zero_spatial_edges(noise_rem)

        noise_rem = noise_rem.astype(np.uint16)
        diff_dict["Noise removal"] = noise_rem

    if area_size != None:
        mask = (output > area_thr).astype(np.uint8)

        refined_output = np.zeros(mask.shape, dtype=np.float32)

        for k in range(len(data)):
            nb_components, out, stats, centroids = \
                cv.connectedComponentsWithStats(mask[k, :, :, None], connectivity=8)


            for i in range(1, nb_components):
                if stats[i, cv.CC_STAT_AREA] >= area_size:
                    refined_output[k][out == i] = output[k][out == i]

        refined_output = zero_spatial_edges(refined_output)
        refined_output = refined_output.astype(np.uint16)
        diff_dict["Area filter"] = refined_output

    diff_dict["Target"] =  target
    for i in range(show_n):
        diffs = {k: v[i] for k, v in diff_dict.items()}
        show_diffractograms(diffs, clip_max)

    if area_size != None:
        return refined_output
    
    return noise_rem

def save_results(results, output_dir):
    output_dir = Path(output_dir)
    new_input = output_dir / "x_input.h5"
    new_target = output_dir / "x_target.h5"
    orig_data = Path(f"dataset/x_input.h5")

    output_dir.mkdir(parents=True, exist_ok=True)


    if not new_input.exists():
        new_input.symlink_to(".." / orig_data)

    with h5py.File(new_target, 'w') as hf:
        for name, data in results.items():

            chunk_shape = (1, *data.shape[1:])
            
            hf.create_dataset(
                name, 
                data=data,  
                compression="gzip", 
                chunks=chunk_shape,
                shuffle=True 
            )