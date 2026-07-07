import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from idiff.bkg2d import gaussian
from unet.dataset_enhancement import zero_spatial_edges
from unet.data import rescale


dataset_dir = Path("unet/dataset/")

# Options for visualization and testing
skip_rest = False # Skips images that are not shown
save_data = True # Controls if results are saved
scale_factor = 1 # Optional target resizing
show_n = 0 # Number of examples to show for each sample


def save_dataset(n, d, result_path):
    if not save_data:
        return
    
    with h5py.File(result_path, 'r+') as f:
        chunk_shape = (1, *d.shape[1:])
        f.create_dataset(
                n, 
                data=d,  
                compression="gzip", 
                chunks=chunk_shape,
                shuffle=True 
            )
        
def enhance_single(img, sigma, thr, area_size, normalize, scale_factor):
        """Process a single image."""
        if scale_factor != 1:
            img_rescaled = rescale(img, scale_factor)
        else:
            img_rescaled = img
        img_rescaled = img_rescaled.astype(np.float32) # This improves processing speed
        img_processed = gaussian(img_rescaled, thr, area_size, sigma, normalize)
        img_processed = zero_spatial_edges(img_processed, border_width=5)
        return img_processed.astype(np.float16)
        
def enhance_fn_mc(dataset, sigma, thr, area_size, normalize, show_n=0, skip_rest=False):
    imgs = [None] * len(dataset)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(enhance_single, dataset[i], sigma, thr, area_size, normalize, scale_factor): i 
                   for i in range(len(dataset)) if not skip_rest or i < show_n}
        
        for future in as_completed(futures):
            i = futures[future]
            imgs[i] = future.result()
        
    for i in range(show_n):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(np.log10(dataset[i] + 1), )
        axs[1].imshow(np.log10(imgs[i] + 1))
        plt.show()

    if not skip_rest:
        return np.stack(imgs)
    else:
        return np.zeros(())
    
if __name__ == "__main__":
    for split_name in ["train", "val"]:
        result_path = dataset_dir / f"{split_name}_target.h5"
        dataset = h5py.File(dataset_dir / f"{split_name}.h5", 'r')

        # Create file
        if save_data:
            with h5py.File(result_path, 'w') as f:
                pass

        name = "au"
        result = enhance_fn_mc(dataset[name], sigma=1.5, thr=6, area_size=10, normalize=True, show_n=show_n, skip_rest=skip_rest)
        save_dataset(name, result, result_path)
        print(f"{name}: {result.shape}")
        result = None # Free memory

        name = "tbf3"
        result = enhance_fn_mc(dataset[name], sigma=10, thr=1, area_size=4, normalize=True, show_n=show_n, skip_rest=skip_rest)
        save_dataset(name, result, result_path)
        print(f"{name}: {result.shape}")
        result = None # Free memory

        name = "feo"
        result = enhance_fn_mc(dataset[name], sigma=3, thr=6, area_size=4, normalize=True, show_n=show_n, skip_rest=skip_rest)
        save_dataset(name, result, result_path)
        print(f"{name}: {result.shape}")
        result = None # Free memory

        name = "laf3"
        result = enhance_fn_mc(dataset[name], sigma=2, thr=6, area_size=5, normalize=True, show_n=show_n, skip_rest=skip_rest)
        save_dataset(name, result, result_path)
        print(f"{name}: {result.shape}")
        result = None # Free memory

        name = "gdf3"
        result = enhance_fn_mc(dataset[name], sigma=10, thr=1, area_size=3, normalize=True, show_n=show_n, skip_rest=skip_rest)
        save_dataset(name, result, result_path)
        print(f"{name}: {result.shape}")
        result = None # Free memory