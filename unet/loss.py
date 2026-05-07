import torch
import torch.nn.functional as F
import numpy as np

def profile_1d_loss(input2d: torch.Tensor, target: tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    intensity, target1d = prepare_profiles(input2d, target)
    return torch.nn.functional.l1_loss(intensity, target1d)

def prepare_profiles(input2d, target) -> tuple[torch.Tensor, torch.Tensor]:
    device = input2d.device
    summed_input2d = sum_aligned_images(input2d)
    # summed_input2d = F.interpolate(summed_input2d.unsqueeze(0), (1024, 1024), mode="bicubic")
    radial_distance, intensity = calc_radial_distribution(summed_input2d.squeeze())

    # remove center and normalize
    intensity[:9] = 0
    intensity = intensity / intensity.max()

    q, I = target
    q, I = q[0].to(device, non_blocking=True), I[0].to(device, non_blocking=True)
    max_target = q[I.argmax()]
    max_input = torch.argmax(intensity)
    # calibration_constant = max_xrd/max_eld
    calibration_constant = max_target/max_input
    target1d = resize_target(q, I, calibration_constant)

    target1d = torch.nn.functional.pad(target1d, (0, intensity.shape[0] - target1d.shape[0]))
    return intensity, target1d


def resize_target(q, I, calibration_constant) -> torch.Tensor:
    # 1. Define the number of bins (15 targets means 16 bin edges)
    num_bins = torch.ceil(q[-1] / calibration_constant).int()
    bin_edges = torch.linspace(0, q.max(), num_bins + 1, device=q.device)

    # 2. Assign each row's float position to a bin (0 to 14)
    bin_indices = torch.bucketize(q, bin_edges)
    # Ensure indices stay within [0, 14]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # 3. Aggregate the values for each bin
    tensor_data = torch.zeros(num_bins, device=q.device)
    for i in range(num_bins):
        mask = bin_indices == i
        if torch.any(mask):
            tensor_data[i] = torch.tensor(I[mask].max(), device=q.device)

    return tensor_data

def sum_aligned_images(images: torch.Tensor) -> torch.Tensor:
    """
    Sums 2D images in a torch tensor of shape (b, 1, x, x) by aligning their centers 
    (the location of the maximum value). Edges are filled with zeros instead of rolling.

    Args:
        images (torch.Tensor): Input tensor of shape (b, 1, x, x).

    Returns:
        torch.Tensor: The summed 2D image of shape (1, x, x).
    """
    if images.dim() != 4 or images.shape[1] != 1:
        raise ValueError("Input tensor must have shape (b, 1, x, x)")
    
    b, _, h, w = images.shape
    
    # Flatten spatial dimensions to find the flat index of the maximum value
    flat_images = images.view(b, -1)
    max_indices = torch.argmax(flat_images, dim=1)
    
    center_x = max_indices // w
    center_y = max_indices % w
    
    target_x = h // 2
    target_y = w // 2
    
    # Calculate shifts
    shift_x = target_x - center_x
    shift_y = target_y - center_y
    
    aligned_sum = torch.zeros((1, h, w), device=images.device)
    
    for i in range(b):
        sx = shift_x[i].item()
        sy = shift_y[i].item()
        
        # Determine the slices for the input image and the target aligned tensor
        # src_x_start/end define the region of the image we read from
        src_x_start = max(0, -sx)
        src_x_end = min(h, h - sx)
        
        src_y_start = max(0, -sy)
        src_y_end = min(w, w - sy)
        
        # dest_x_start/end define where the extracted region goes in the aligned output
        dest_x_start = max(0, sx)
        dest_x_end = min(h, h + sx)
        
        dest_y_start = max(0, sy)
        dest_y_end = min(w, w + sy)
        
        # Extract and paste the window while the edges are implicitly zeroed
        aligned_sum[0, dest_x_start:dest_x_end, dest_y_start:dest_y_end] += \
            images[i, 0, src_x_start:src_x_end, src_y_start:src_y_end]
            
    return aligned_sum

def calc_radial_distribution(tensor: torch.Tensor, center: tuple = None, max_radius: float = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate 1D radially averaged distribution profile from a 2D diffraction pattern.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The 2D tensor which contains the 2D-diffractogram (assumed to be centered).
    center : tuple of two floats, optional, default is None
        The coordinates (xc, yc) of the center. If None, it assumes the center 
        of the tensor.
    max_radius : float, optional
        The maximum radius to calculate up to. If None, uses the distance 
        from the center to the furthest corner.
        
    Returns
    -------
    radial_distance : torch.Tensor
        1D tensor of radii distances.
    intensity : torch.Tensor
        1D tensor of average intensities corresponding to the radii.
    """
    height, width = tensor.shape
    
    # Use the specified center, or default to the center of the image
    if center is None:
        xc, yc = width / 2.0, height / 2.0
    else:
        xc, yc = center
        
    # Create coordinate grids and calculate the distance (R) tensor
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=tensor.device, dtype=tensor.dtype),
        torch.arange(width, device=tensor.device, dtype=tensor.dtype),
        indexing='ij'
    )
    
    X = x_coords - xc
    Y = y_coords - yc
    R = torch.sqrt(X**2 + Y**2)
    
    if max_radius is None:
        max_radius = torch.max(R).item()
        
    radial_distance = torch.arange(1, int(max_radius), device=tensor.device, dtype=tensor.dtype)
    intensity = torch.zeros_like(radial_distance, dtype=tensor.dtype)
    
    bin_size = 1.0
    
    # Compute the mean intensity for each radial distance bin
    for i, r in enumerate(radial_distance):
        mask = (R > (r - bin_size)) & (R < (r + bin_size))
        
        if mask.any():
            intensity[i] = tensor[mask].mean()
            
    return radial_distance, intensity