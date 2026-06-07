import torch
import torch.nn.functional as F
import numpy as np
import ediff

class CombinedLoss(torch.nn.Module):
    def __init__(self, logspace=False, profile_scale=1):
        super().__init__()
        self.logspace = logspace
        self.profile_scale = profile_scale

    def forward(self, x: torch.Tensor, y, a=1, b=1) -> torch.Tensor:
        device = x.device
        y, p = y
        y = y.to(device, non_blocking=True)
        if self.logspace:
            return a * torch.nn.functional.huber_loss(x, torch.log1p(y)) + \
                b * (profile_1d_loss(torch.expm1(x), p, self.profile_scale) if b > 0 else 0)
        else:
            return a * torch.nn.functional.huber_loss(x, y) + \
                b * (profile_1d_loss(x, p, self.profile_scale) if b > 0 else 0)

def profile_1d_loss(input2d: torch.Tensor, target, profile_scale=1) -> torch.Tensor:
    intensity, target1d = prepare_profiles(input2d, target, profile_scale)
    return torch.nn.functional.huber_loss(intensity, target1d)

def prepare_profiles(input2d, target, profile_scale=1) -> tuple[torch.Tensor, torch.Tensor]:
    device = input2d.device
    summed_input2d = sum_aligned_images(input2d)
    if profile_scale != 1:
        summed_input2d = F.interpolate(summed_input2d.unsqueeze(0), scale_factor=profile_scale, mode="bicubic")
    radial_distance, intensity = calc_radial_distribution(summed_input2d.squeeze())

    # Batch contains target for each input, however, all of them should be identical
    q, I, center_size, calibration_constant = target
    center_size = center_size[0] * profile_scale
    calibration_constant = calibration_constant[0] * profile_scale
    q, I = q[0].to(device, non_blocking=True), I[0].to(device, non_blocking=True)

    # remove center and normalize
    intensity[:center_size] = 0
    # Biggest peak should be in the first half
    # (helps avoid high values around the edges)
    intensity = intensity / intensity[:intensity.shape[0] // 2].max()

    target1d = resize_target(q, I, calibration_constant)

    target1d = torch.nn.functional.pad(target1d, (0, intensity.shape[0] - target1d.shape[0]))
    return intensity, target1d

def nearest_interpolate_1d(x, y, M):
    """
    Interpolates values from y (defined at coordinates x) 
    onto a uniform grid of size M (coordinates 0 to M-1) 
    using nearest-neighbor interpolation.
    """
    # 1. Create the implicit coordinates for the target tensor z
    # shape: (M,) -> values: [0, 1, 2, ..., M-1]
    z_coords = torch.arange(M, dtype=x.dtype, device=x.device)
    
    # 2. Compute absolute distances between every target coordinate and every source coordinate x
    # We use broadcasting: (M, 1) - (1, N) -> (M, N)
    distances = torch.abs(z_coords.unsqueeze(1) - x.unsqueeze(0))
    
    # 3. For each target coordinate, find the index of the nearest source coordinate in x
    # shape: (M,)
    nearest_indices = torch.argmin(distances, dim=1)
    
    # 4. Gather the corresponding values from y
    z = y[nearest_indices]
    
    return z

def resize_target(q, I, calibration_constant, nearest=False) -> torch.Tensor:
    # 0. Define the number of bins
    N = torch.ceil(q[-1] * calibration_constant).int()

    if nearest:
        return nearest_interpolate_1d(q * calibration_constant, I, N)

    # 1. Round x and convert to long/int so it can be used as indices
    # We also clip the values to ensure they stay within [0, N-1]
    indices = torch.round(q * calibration_constant).long().clamp(0, N - 1)

    # 2. Initialize the output tensor of length N
    out = torch.zeros((N,), dtype=I.dtype, device=I.device)

    # 3. Use scatter_reduce to find the maximum for each index
    out.scatter_reduce_(dim=0, index=indices, src=I, reduce="amax", include_self=False)

    return out

def sum_aligned_images(images: torch.Tensor, centers=None) -> torch.Tensor:
    """
    Sums 2D images in a torch tensor of shape (b, 1, x, x) by aligning their centers 
    (the location of the maximum value). Edges are filled with zeros instead of rolling.

    Args:
        images (torch.Tensor): Input tensor of shape (b, 1, x, x).
        centers (torch.Tensor): Centres of images - shape (b, 2).

    Returns:
        torch.Tensor: The summed 2D image of shape (1, x, x).
    """
    if images.dim() != 4 or images.shape[1] != 1:
        raise ValueError("Input tensor must have shape (b, 1, x, x)")
    
    b, _, h, w = images.shape
    
    target_x = h // 2
    target_y = w // 2
    
    aligned_sum = torch.zeros((1, h, w), device=images.device)
    
    center_locator = ediff.center.IntensityCenter()
    for i in range(b):
        if centers is None:
            csquare = max(20, h // 10)
            cx, cy = center_locator.center_of_intensity(images[i, 0].detach().cpu().numpy(), csquare, 0.8)
        else:
            cx, cy = centers[i]
        if np.isfinite(cx) and np.isfinite(cy):
            cx, cy = round(cx), round(cy)
            sx = target_x - cx
            sy = target_y - cy
        else:
            sx, sy = 0, 0
        
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