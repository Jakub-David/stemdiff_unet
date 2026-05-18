import numpy as np
from plot import show_1D_profiles, show_diffractograms
import torch
from pathlib import Path
import h5py
import cv2 as cv
from scipy.ndimage import white_tophat

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

def get_disk_footprint(radius, dim3=True):
    """
    Creates a 3D disk footprint for a stack of images.
    Returns a (1, height, width) bool array.
    """
    # Create a coordinate grid from -radius to +radius
    # (2*radius + 1) dimensions total
    y, x = np.ogrid[-radius : radius+1, -radius : radius+1]
    
    # Mathematical definition of a disk: x^2 + y^2 <= r^2
    disk_2d = x**2 + y**2 <= radius**2
    
    if dim3:
        # Reshape for 3D stack (1, height, width)
        # Cast explicitly to bool for safety/efficiency
        return disk_2d[np.newaxis, :, :].astype(bool)
    else:
        return disk_2d.astype(bool)

def remove_small_components(x, area_size=3, area_thr=1, area_con=4):
    mask = (x > area_thr).astype(np.uint8)

    refined_output = np.zeros_like(x)

    nb_components, out, stats, centroids = \
        cv.connectedComponentsWithStats(
            mask[:, :, None], 
            connectivity=area_con
        )


    for i in range(1, nb_components):
        if stats[i, cv.CC_STAT_AREA] >= area_size:
            refined_output[out == i] = x[out == i]

    return refined_output

def process(name, model=None, thr=None, area_size=None, area_thr=1, area_con=8,
            show_n=5, clip_max=1, tophat_ker=3, show_profiles=False, 
            limit=None):
    # Load
    with h5py.File("dataset/x_input.h5", 'r') as f_in:
        data = f_in[f"input{name}"][:]
    
    # if model != None:
    with h5py.File("dataset/x_target.h5", 'r') as f_tar:
        target = f_tar[f"target{name}"][:]

    if limit != None:
        data = data[:limit]
        target = target[:limit]

    diff_dict = { "Original": data }

    # Enhance with unet
    if model != None:
        output = predict(model, data)
        diff_dict["Result"] = output
    else:
        footprint = get_disk_footprint(tophat_ker)
        output = white_tophat(data, footprint=footprint)
        diff_dict["Tophat"] = output


    # Enhance output
    noise_rem = output.copy()
    if thr != None:
        noise_rem[noise_rem<thr]=0
        noise_rem = zero_spatial_edges(noise_rem)

        noise_rem = noise_rem.astype(np.uint16)
        diff_dict["Noise removal"] = noise_rem

    if area_size != None:
        refined_output = np.zeros_like(output)

        for i in range(len(data)):
            refined_output[i] = remove_small_components(
                output[i], area_size, area_thr, area_con)


        refined_output = zero_spatial_edges(refined_output)
        refined_output = refined_output.astype(np.uint16)
        diff_dict["Area filter"] = refined_output

    # if model != None: 
    diff_dict["Target"] =  target

    for i in range(show_n):
        diffs = {k: v[i] for k, v in diff_dict.items()}
        show_diffractograms(diffs, clip_max)
        if show_profiles:
            show_1D_profiles(diffs)

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

    save_h5(results, new_target)

def save_h5(data, path, compression="gzip"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as hf:
        for name, data in data.items():

            chunk_shape = (1, *data.shape[1:])
            
            hf.create_dataset(
                name, 
                data=data,  
                compression=compression, 
                chunks=chunk_shape,
                shuffle=True 
            )