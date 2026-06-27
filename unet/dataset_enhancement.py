from pathlib import Path
import h5py

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