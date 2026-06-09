import torch
import torch.nn.functional as F
import numpy as np
import ediff

class CombinedLoss(torch.nn.Module):
    def __init__(self, device, logspace=False, profile_scale=1, individual_profiles=False, include_2d=True):
        super().__init__()
        self.logspace = logspace
        self.profile_scale = profile_scale
        self.individual_profiles = individual_profiles
        self.rad_dist = RadialDistribution(256 * profile_scale, 256 * profile_scale, device)
        self.include_2d = include_2d

    def forward(self, x: torch.Tensor, y, a=1, b=1, return_parts=False) -> torch.Tensor:
        if self.include_2d:
            p = y[1:]
            y = y[0]
        else:
            p = y

        if self.include_2d:
            if self.logspace:
                y = torch.log1p(y)

            loss_2d = torch.nn.functional.huber_loss(x, y)
        else:
            loss_2d = 0
            
        if b > 0:
            if self.logspace:
                x = torch.expm1(x)

            x1d, target1d = prepare_profiles(x, p, self.individual_profiles, self.rad_dist, self.profile_scale)
            loss_1d = torch.nn.functional.huber_loss(x1d, target1d)
        else:
            loss_1d = 0

        if return_parts:
            return a * loss_2d + b * loss_1d, loss_2d, loss_1d
        else:
            return a * loss_2d + b * loss_1d

def prepare_profiles(input2d, target, individual_profiles, rad_dist, profile_scale=1) -> tuple[torch.Tensor, torch.Tensor]:
    target_profile, center_sizes, centers = target
    centered_input2d = center_images(input2d, centers).squeeze()

    if individual_profiles:
        # 1. Batch Interpolation: (B, H, W) -> (B, 1, H, W) -> Interpolate -> (B, H_new, W_new)
        if profile_scale != 1:
            centered_input2d = F.interpolate(
                centered_input2d.unsqueeze(1), 
                scale_factor=profile_scale, 
                mode="bicubic"
            ).squeeze(1)

        # 2. Batch Radial Distance: Expects (B, H, W), returns (B, seq_len)
        _, intensity = rad_dist(centered_input2d)
        
        # 3. Vectorized Masking: Create a 2D mask matching intensity shape (B, seq_len)
        # arange shape: (1, seq_len)
        steps = torch.arange(intensity.shape[1], device=intensity.device).unsqueeze(0)
        # effective_centers shape: (B, 1)
        effective_centers = (center_sizes * profile_scale).unsqueeze(1)
        
        # Mask out values where the index is less than the center size
        intensity[steps < effective_centers] = 0

        # 4. Vectorized Normalization: Max over the first half for each row
        half_len = intensity.shape[1] // 2
        # Keepdim=True ensures max_vals shape is (B, 1) for proper broadcasting
        max_vals, _ = intensity[:, :half_len].max(dim=1, keepdim=True)
        
        # Prevent division by zero if an entire row is zero
        max_vals = torch.clamp(max_vals, min=1e-8)
        intensity = intensity / max_vals

        # 5. Vectorized Padding for target profiles
        pad_len = intensity.shape[1] - target_profile.shape[1]
        target_profile = F.pad(target_profile, (0, pad_len))

        return intensity, target_profile
    else:
        centered_input2d = centered_input2d.sum(dim=0)

        if profile_scale != 1:
            centered_input2d = F.interpolate(centered_input2d.unsqueeze(0), scale_factor=profile_scale, mode="bicubic")
            centered_input2d = centered_input2d.squeeze()

        radial_distance, intensity = rad_dist(centered_input2d)

        # Batch contains target for each input, however, all of them should be identical here
        center_size = center_sizes[0]
        target_profile = target_profile[0]

        # remove center and normalize
        intensity[:center_size * profile_scale] = 0
        # Biggest peak should be in the first half
        # (helps avoid high values around the edges)
        intensity = intensity / intensity[:intensity.shape[0] // 2].max()

        target_profile = torch.nn.functional.pad(target_profile, (0, intensity.shape[0] - target_profile.shape[0]))
        return intensity, target_profile

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
    N = torch.round(q[-1] * calibration_constant).int()

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

def center_images(images: torch.Tensor, centers=None) -> torch.Tensor:
    """
    Center 2D images in a torch tensor of shape (b, 1, x, x).
    Edges are filled with zeros. Fully vectorized using grid_sample.
    """
    if images.dim() != 4 or images.shape[1] != 1:
        raise ValueError("Input tensor must have shape (b, 1, x, x)")
    
    b, c, h, w = images.shape
    device = images.device
    
    target_x = h // 2
    target_y = w // 2
    
    # --- Step 1: Compute Centers for the Entire Batch ---
    if centers is None:
        centers = np.zeros((b, 2))
        center_locator = ediff.center.IntensityCenter()
        csquare = max(20, h // 10)
        for i in range(b):
            cx, cy = center_locator.center_of_intensity(images[i, 0].detach().cpu().numpy(), csquare, 0.8)
            centers[i] = [cx, cy] if (np.isfinite(cx) and np.isfinite(cy)) else [target_x, target_y]
        centers = torch.tensor(centers, dtype=torch.float32, device=device)
    else:
        # Ensure centers is a float tensor on the correct device
        if not isinstance(centers, torch.Tensor):
            centers = torch.tensor(centers, dtype=torch.float32, device=device)
        else:
            centers = centers.to(device=device, dtype=torch.float32)
            
        # Handle any NaN/Inf values globally
        invalid_mask = ~torch.isfinite(centers).all(dim=1)
        centers[invalid_mask] = torch.tensor([target_x, target_y], dtype=torch.float32, device=device)

    # --- Step 2: Calculate Shifts and Normalize to Grid Coordinates ---
    # grid_sample expects coordinates normalized between -1 and 1.
    # A shift of 1 pixel in x corresponds to 2 / (w - 1) in grid space.
    
    # Round centers to match your original pixel-snapping logic
    centers = torch.round(centers)
    
    # Calculate the pixel displacement (shift) needed
    shift_x = target_x - centers[:, 0]
    shift_y = target_y - centers[:, 1]
    
    # Convert pixel shifts to grid_sample's normalized scale [-1, 1]
    # Note: grid_sample expects (delta_y, delta_x) or (delta_col, delta_row) format
    norm_shift_x = -shift_x * (2.0 / (h - 1))
    norm_shift_y = -shift_y * (2.0 / (w - 1))
    
    # --- Step 3: Create the Affine Transformation Matrix ---
    # We construct a [B, 2, 3] affine matrix for translation:
    # [ 1  0  shift_y ]
    # [ 0  1  shift_x ]
    theta = torch.zeros((b, 2, 3), device=device, dtype=torch.float32)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = norm_shift_y
    theta[:, 1, 2] = norm_shift_x
    
    # --- Step 4: Sample ---
    # Generate the sampling grid based on the transformation matrices
    grid = F.affine_grid(theta, size=images.size(), align_corners=True)
    
    # Remap all images simultaneously. padding_mode="zeros" handles the edges perfectly.
    # mode="nearest" keeps the pixel values discrete.
    aligned_images = F.grid_sample(images, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    
    return aligned_images

class RadialDistribution(torch.nn.Module):
    def __init__(self, height: int, width: int, device: torch.device, dtype=torch.float32):
        super().__init__()
        self.height = height
        self.width = width
        
        # 1. Compute fixed center
        xc, yc = width / 2.0, height / 2.0
        
        # 2. Pre-calculate coordinate grids once
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij'
        )
        
        R = torch.sqrt((x_coords - xc)**2 + (y_coords - yc)**2)
        max_radius = torch.max(R).item()
        self.max_bin_idx = int(max_radius)
        
        # 3. Pre-calculate and cache the radial distance labels
        radial_distance = torch.arange(0, self.max_bin_idx, device=device, dtype=dtype)
        self.register_buffer('radial_distance', radial_distance)
        
        # 4. Pre-calculate and cache the bin indices and masks
        bin_indices = torch.round(R).long()
        valid_mask = (bin_indices >= 0) & (bin_indices < self.max_bin_idx)
        
        # Flattened valid locations
        self.register_buffer('valid_bins_1d', bin_indices[valid_mask])
        self.register_buffer('valid_mask_1d', valid_mask.flatten())
        
        # 5. Pre-calculate pixel counts per bin for a single image
        pixel_counts = torch.bincount(self.valid_bins_1d, minlength=self.max_bin_idx)
        self.register_buffer('pixel_counts', pixel_counts)
        
        # Create a mask for safe division (avoiding division by zero)
        self.register_buffer('safe_division_mask', pixel_counts > 0)

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Supports both 2D tensors (H, W) and 3D tensors (B, H, W).
        """
        if tensor.dim() == 2:
            # --- Single Image Mode ---
            valid_intensities = tensor.flatten()[self.valid_mask_1d]
            sum_intensity = torch.bincount(self.valid_bins_1d, weights=valid_intensities, minlength=self.max_bin_idx)
            
            intensity = torch.where(
                self.safe_division_mask, 
                sum_intensity / self.pixel_counts, 
                torch.zeros_like(sum_intensity)
            )
            return self.radial_distance, intensity

        elif tensor.dim() == 3:
            # --- Batch Mode (B, H, W) ---
            b, h, w = tensor.shape
            
            # 1. Expand the valid mask to cover the entire batch and flatten the tensor
            # valid_mask_batch becomes a boolean mask of shape (B * H * W)
            valid_mask_batch = self.valid_mask_1d.repeat(b)
            flat_tensor = tensor.flatten()
            valid_intensities = flat_tensor[valid_mask_batch]
            
            # 2. Shift bin indices for each batch item so they don't overlap in bincount
            # e.g., Batch 0 uses bins [0, max_bin), Batch 1 uses [max_bin, 2*max_bin), etc.
            batch_offsets = torch.arange(b, device=tensor.device).view(b, 1) * self.max_bin_idx
            
            # Filter out the valid bins, inject the offsets, and flatten to 1D
            # valid_bins_1d is shape (V,), batch_offsets is (B, 1). Broadcasting handles it perfectly.
            shifted_bins = (self.valid_bins_1d + batch_offsets).flatten()
            
            # 3. Perform a single bincount across the entire batch
            total_bins = b * self.max_bin_idx
            sum_intensity_flat = torch.bincount(shifted_bins, weights=valid_intensities, minlength=total_bins)
            
            # 4. Reshape back to (B, max_bin_idx)
            sum_intensity = sum_intensity_flat.view(b, self.max_bin_idx)
            
            # 5. Compute vectorized batch means safely using broadcasting
            # pixel_counts is (max_bin_idx,), broadcasted to match (B, max_bin_idx)
            intensity = torch.where(
                self.safe_division_mask,
                sum_intensity / self.pixel_counts,
                torch.zeros_like(sum_intensity)
            )
            
            return self.radial_distance, intensity
        
        else:
            raise ValueError("Input tensor must be 2D (H, W) or 3D (B, H, W)")