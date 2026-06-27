import torch
from loss import CombinedLoss, ReverseKLDivLoss

def psnr(pred, target, max_val=11810.0):
    """
    Compute Peak Signal-to-Noise Ratio
    pred, target: tensors [B, C, H, W]
    max_val: maximum possible pixel value
    """
    mse = torch.mean((pred - target) ** 2, dim=[-2, -1])
    return torch.where(
        mse == 0, 
        float("inf"), 
        20 * torch.log10(max_val / torch.sqrt(mse))
    ).squeeze()

def entropy(img_tensor):
    discrete_img = torch.round(img_tensor).long()
    
    hist = torch.bincount(discrete_img.flatten(), minlength=12000).float()
    
    probs = hist / hist.sum()
    entropy = torch.sum(torch.special.entr(probs)) / torch.log(torch.tensor(2.0))
    return entropy

def calculate_batch_entropy(x, y):
    x_entropy = torch.stack([entropy(img) for img in x])
    y_entropy = torch.stack([entropy(img) for img in y])
    entropy_delta = y_entropy - x_entropy
    return y_entropy, entropy_delta

@torch.no_grad()
def evaluate(model, loader, device, criterion=None, return_every=0, a=1, b=1):
    """
    Evaluate model on DataLoader.
    
    Args:
        model: your ResidualUNet
        loader: DataLoader
        device: torch device
        criterion: loss function (optional)
        return_examples: number of example predictions to return
        
    Returns:
        avg_loss: average loss over dataset (if criterion given)
        avg_psnr: average PSNR
        examples: list of tuples (input, clean_pred, target) for inspection
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_loss_1d = 0.0
    total_loss_2d = 0.0
    total_entropy = 0.0
    total_entropy_delta = 0.0
    total_rkl = 0.0
    total_images = 0
    total_batches = 0
    examples = []

    rkl = ReverseKLDivLoss()

    # ensure 1d profile is calculated
    if b == 0:
        b = 1e-20

    for data_dict in loader:
        data_dict = {n: t.to(device, non_blocking=True) for n, t in data_dict.items()}

        clean_pred = model(data_dict["original_image"])

        if isinstance(criterion, CombinedLoss):
            if model.logspace:
                loss, loss_2d, loss_1d, clean1d, target1d = \
                    criterion(clean_pred.log1p(), data_dict, a, b, return_parts=True)
            else:
                loss, loss_2d, loss_1d, clean1d, target1d = \
                    criterion(clean_pred, data_dict, a, b, return_parts=True)
            total_loss += loss
            if isinstance(loss_1d, torch.Tensor):
                total_loss_1d += loss_1d
            if isinstance(loss_2d, torch.Tensor):
                total_loss_2d += loss_2d
            
            total_rkl += rkl(clean1d, target1d)
        elif criterion is not None:
            if model.logspace:
                loss = criterion(clean_pred.log1p(), data_dict["target_2d"].log1p())
            else:
                loss = criterion(clean_pred, data_dict["target_2d"])
            total_loss += loss


        # Compute PSNR and entropy per batch
        if "target_2d" in data_dict:
            batch_psnr = psnr(clean_pred, data_dict["target_2d"])
            total_psnr += batch_psnr.sum()

        batch_entropy, batch_entropy_delta = calculate_batch_entropy(data_dict["original_image"], clean_pred)
        total_entropy += batch_entropy.sum()
        total_entropy_delta += batch_entropy_delta.sum()

        # Save some examples if requested
        if return_every > 0 and total_batches % return_every == 0:
            examples.append((
                data_dict["original_image"][:4].to('cpu', non_blocking=True),
                clean_pred[:4].to('cpu', non_blocking=True),
                data_dict["target_2d"][:4].to('cpu', non_blocking=True),
                clean1d.to('cpu', non_blocking=True),
                target1d.to('cpu', non_blocking=True)
            ))

        total_images += data_dict["original_image"].shape[0]
        total_batches += 1

    results = {}
    img_pixels = data_dict["original_image"].shape[-2] * data_dict["original_image"].shape[-1]
    if "target_2d" in data_dict:
        results["avg_psnr"] = total_psnr.item() / total_images 
    if isinstance(criterion, CombinedLoss):
        if criterion.loss_1d is not None:
            results[f"avg_1d_{criterion.loss_1d.__class__.__name__}"] = total_loss_1d.item() / total_images 
        if criterion.loss_2d is not None:
            results[f"avg_2d_{criterion.loss_2d.__class__.__name__}"] = total_loss_2d.item() / total_images
        if criterion.loss_1d is not None and criterion.loss_2d is not None:
            results["avg_loss"] = (
                a * (total_loss_2d.item() / img_pixels) + 
                b * (total_loss_1d.item() / target1d.shape[-1])
            ) / total_images
    elif criterion is not None:
        results["avg_loss"] = (total_loss.item() / total_images) / img_pixels

    results["reverse_kl_div"] = total_rkl.item() / total_batches # assume 1 profile per batch

    results["output_entropy"] = total_entropy.item() / total_images
    results["entropy_delta"] = total_entropy_delta.item() / total_images

    return results, examples