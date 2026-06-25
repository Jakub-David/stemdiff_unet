import torch
import torch.nn.functional as F
import torchmetrics
import numpy as np
import ediff

class ReverseKLDivLoss(torch.nn.Module):
    def __init__(self, epsilon = 1e-12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")
        self.epsilon = epsilon
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Ensure tensors have batch dimension
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if prediction.ndim == 1:
            prediction = prediction.unsqueeze(0)

        # 2. Smooth tensors
        target = target + self.epsilon
        prediction = prediction + self.epsilon

        # 3. Normalize
        target = target / target.sum(dim=-1, keepdim=True)
        prediction = prediction / prediction.sum(dim=-1, keepdim=True)

        # 4. Compute KL Divergence with reversed arguments
        return self.kl_div(target.log(), prediction)
    
class SymmetricMAPELoss(torch.nn.Module):
    def __init__(self, epsilon = 1e-12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate loss
        numerator = torch.abs(prediction - target)
        denominator = prediction.abs() + target.abs() + self.epsilon
        return 2 * torch.mean(numerator / denominator)


class CombinedLoss(torch.nn.Module):
    def __init__(self, device, logspace=False, profile_scale=1, individual_profiles=False, 
                 loss_1d=None, loss_2d=None, l1_reg=0, total_variation=0, local_cons_reg=0):
        super().__init__()
        self.logspace = logspace
        self.profile_scale = profile_scale
        self.individual_profiles = individual_profiles
        self.rad_dist = RadialDistribution(256 * profile_scale, 256 * profile_scale, device)
        self.loss_2d = loss_2d
        self.loss_1d = loss_1d
        self.l1_reg_w = l1_reg
        self.total_variation_w = total_variation
        self.local_cons_reg_w = local_cons_reg
        if local_cons_reg > 0:

            k_s_size = 3
            k_s_2d = gaussian_kernel_2d(k_s_size, 1, device)
            self.conv_s = torch.nn.Conv2d(1, 1, kernel_size=k_s_size, padding="same", bias=False, device=device)
            self.conv_s.weight.data = k_s_2d.view(1, 1, k_s_size, k_s_size)

            k_l_size = 4
            self.border_width = k_l_size + 3
            k_l_2d = gaussian_kernel_2d(k_l_size, 2, device)
            # k_l_2d = torch.ones((k_l_size, k_l_size), dtype=torch.float32, device=device) / (k_l_size ** 2)
            self.conv_l = torch.nn.Conv2d(1, 1, kernel_size=k_l_size, dilation=2, padding="same", bias=False, device=device)
            self.conv_l.weight.data = k_l_2d.view(1, 1, k_l_size, k_l_size)

    def forward(self, orig: torch.Tensor, clean: torch.Tensor, y, a=1, b=1, return_parts=False) -> torch.Tensor:
        if len(y) == 4:
            p = y[1:]
            y = y[0]
        else:
            p = y
            y = None

        if self.logspace and isinstance(y, torch.Tensor):
            y = torch.log1p(y)

        if self.loss_2d is not None and a > 0:
            loss_2d = self.loss_2d(clean, y)
        else:
            loss_2d = 0
            
        if self.loss_1d is not None and b > 0:
            if self.logspace:
                x_exp = torch.expm1(clean)
            else:
                x_exp = clean

            x1d, target1d = prepare_profiles(x_exp, p, self.individual_profiles, self.rad_dist, self.profile_scale)
            loss_1d = self.loss_1d(x1d, target1d)
        else:
            x1d, target1d = None, None
            loss_1d = 0

        if self.l1_reg_w > 0:
            # l1 norm + negative penalty
            l1_reg = torch.nn.functional.huber_loss(
                clean,
                torch.zeros_like(clean)
            ) + 10 * torch.mean(torch.relu(-clean))
        else:
            l1_reg = 0

        if self.total_variation_w > 0:
            # use mean reduction to keep scale closer to other losses
            tv = torchmetrics.functional.total_variation(clean, reduction="mean")
        else:
            tv = 0

        if self.local_cons_reg_w > 0:
            if self.logspace:
                orig = torch.log1p(orig)

            bw = self.border_width

            # Calculate means
            means_clean_s = self.conv_s(clean)
            means_orig_s = self.conv_s(orig)

            means_clean_l = self.conv_l(clean)
            means_orig_l = self.conv_l(orig)

            # Compute absolute differences between features and remove borders
            diff_clean = (means_clean_l - means_clean_s)[..., bw:-bw, bw:-bw]
            diff_orig = (means_orig_l - means_orig_s)[..., bw:-bw, bw:-bw]

            # Compute loss
            error = diff_clean - diff_orig
            lc_reg = torch.nn.functional.huber_loss(
                error, 
                torch.zeros_like(error), 
            )
        else:
            lc_reg = 0

        final_loss = a * loss_2d + \
                     b * loss_1d + \
                     self.l1_reg_w * l1_reg + \
                     self.total_variation_w * tv + \
                     self.local_cons_reg_w * lc_reg

        if return_parts:
            return final_loss, loss_2d, loss_1d, x1d, target1d
        else:
            return final_loss

def prepare_profiles(input2d, target, individual_profiles, rad_dist, profile_scale=1) -> tuple[torch.Tensor, torch.Tensor]:
    target_profile, center_sizes, centers = target
    
    # --- Step 1: Center Images ---
    centered_input2d = center_images(input2d, centers).squeeze(1)  # Shape: (B, H, W)

    # --- Step 2: Handle Aggregation if not individual ---
    if not individual_profiles:
        # Sum across the batch dimension to create a single global image
        # Keepdim=True preserves the batch dimension as 1: Shape (1, H, W)
        centered_input2d = centered_input2d.sum(dim=0, keepdim=True)
        # For target profiles and center sizes, we only care about the first one
        center_sizes = center_sizes[0:1]
        target_profile = target_profile[0:1]

    # --- Step 3: Batch Interpolation ---
    if profile_scale != 1:
        centered_input2d = F.interpolate(
            centered_input2d.unsqueeze(1), 
            scale_factor=profile_scale, 
            mode="bicubic"
        ).squeeze(1)

    # --- Step 4: Radial Distance Extraction ---
    # radial_distance shape: (seq_len,)
    # intensity shape: (B, seq_len)
    radial_distance, intensity = rad_dist(centered_input2d)
    
    # --- Step 5: Center Masking (Using radial_distance View) ---
    # unsqueeze(0) safely creates a view of shape (1, seq_len) without altering the original
    radial_coords = radial_distance.unsqueeze(0) 
    effective_centers = (center_sizes * profile_scale).unsqueeze(1)  # Shape: (B, 1)
    
    # Generate a safe boolean multiplier mask
    mask = (radial_coords >= effective_centers).to(dtype=intensity.dtype)
    intensity = intensity * mask

    # --- Step 6: Normalization with Low-Value Skip Protection ---
    # Max peak should be in the first half (avoids noise around the edges)
    half_len = intensity.shape[1] // 2
    max_vals, _ = intensity[:, :half_len].max(dim=1, keepdim=True)  # Shape: (B, 1)
    
    # Define a strict threshold below which signal is considered non-existent/noise
    min_signal_threshold = 0.1
    
    # Vectorized conditional normalization: 
    # If max_vals > threshold, divide by (max_vals + 1e-6) for smooth scaling.
    # Otherwise, leave the row as zeros to completely bypass zero-division or noise scaling.
    intensity = torch.where(
        max_vals > min_signal_threshold,
        intensity / (max_vals + 1e-6),
        torch.zeros_like(intensity)
    )

    # --- Step 7: Vectorized Target Padding ---
    pad_len = intensity.shape[1] - target_profile.shape[1]
    if pad_len > 0:
        target_profile = F.pad(target_profile, (0, pad_len))
    elif pad_len < 0:
        target_profile = target_profile[:, :intensity.shape[1]]

    # Squeeze dimensions back to (seq_len,) if individual_profiles was False
    if not individual_profiles:
        intensity = intensity.squeeze(0)
        target_profile = target_profile.squeeze(0)

    return intensity, target_profile

def resize_target(q, I, calibration_constant) -> torch.Tensor:
    # 0. Define the number of bins
    N = torch.round(q[-1] * calibration_constant).int()

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
        
        # Cache the 1D spatial mask for flat-mapping
        valid_mask_1d = valid_mask.flatten()
        self.register_buffer('valid_mask_1d', valid_mask_1d)
        
        # Pre-expand valid_bins_1d to 2D shape (1, Num_Valid_Pixels) so it's ready to broadcast with batch size
        valid_bins_1d = bin_indices[valid_mask].unsqueeze(0) 
        self.register_buffer('valid_bins_1d', valid_bins_1d)
        
        # 5. Pre-calculate pixel counts per bin for a single image
        pixel_counts = torch.bincount(valid_bins_1d.flatten(), minlength=self.max_bin_idx)
        self.register_buffer('pixel_counts', pixel_counts)
        
        # Create a mask for safe division (avoiding division by zero)
        self.register_buffer('safe_division_mask', pixel_counts > 0)

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Handles both 2D (H,W) and 3D (B,H,W).
        Everything except batch dimension expansion is entirely precomputed.
        """
        is_2d = tensor.dim() == 2
        if is_2d:
            tensor = tensor.unsqueeze(0)  # (1, H, W)
        elif tensor.dim() != 3:
            raise ValueError("Input tensor must be 2D (H, W) or 3D (B, H, W)")

        b = tensor.shape[0]

        # 1. Flatten spatial dimensions to (B, H*W) and apply the precomputed 1D mask
        # Advanced indexing extracts only valid spatial pixels across all batches simultaneously
        valid_intensities = tensor.view(b, -1)[:, self.valid_mask_1d]  # Shape: (B, Num_Valid_Pixels)

        # 2. Broadcast the precomputed bin indices to match the current batch size
        # expanded_bins becomes shape: (B, Num_Valid_Pixels)
        expanded_bins = self.valid_bins_1d.expand(b, -1)

        # 3. Accumulate sums cleanly using 2D scatter_add_ along dim 1
        sum_intensity = torch.zeros(b, self.max_bin_idx, device=tensor.device, dtype=tensor.dtype)
        sum_intensity.scatter_add_(1, expanded_bins, valid_intensities)

        # 4. Compute means safely using precomputed safe masks and pixel counts
        intensity = torch.where(
            self.safe_division_mask,
            sum_intensity / self.pixel_counts,
            torch.zeros_like(sum_intensity)
        )

        if is_2d:
            intensity = intensity.squeeze(0)

        return self.radial_distance, intensity
    
def gaussian_kernel_2d(kernel_size, sigma, device):
    x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    
    # Normalize it to sum to 1
    gauss_2d = gauss_2d / gauss_2d.sum()

    return gauss_2d