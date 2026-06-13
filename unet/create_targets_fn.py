import numpy as np
from idiff.bkg2d import gaussian
from dataset_enhancement import zero_spatial_edges
from data import rescale

def enhance_single(img, sigma, thr, area_size, scale_factor):
        """Process a single image."""
        if scale_factor != 1:
            img_rescaled = rescale(img, scale_factor)
        else:
            img_rescaled = img
        img_rescaled = img_rescaled.astype(np.float32) # This improves processing speed
        img_processed = gaussian(img_rescaled, thr, area_size, sigma)
        img_processed = zero_spatial_edges(img_processed, border_width=5)
        return img_processed.astype(np.float16)
