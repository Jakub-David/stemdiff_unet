import torch
from loss import CombinedLoss
import numpy as np

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
    x_entropy = np.array([entropy(img).item() for img in x])
    y_entropy = np.array([entropy(img).item() for img in y])
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
    total_images = 0
    examples = []


    for x, t in loader:
        x = x.to(device, non_blocking=True)
        if isinstance(t, torch.Tensor):
            y = t.to(device, non_blocking=True)
        else:
            t = [z.to(device, non_blocking=True) for z in t]
            if isinstance(t[0], torch.Tensor):
                y = t[0]
            else:
                y = None

        clean_pred = model(x)

        if isinstance(criterion, CombinedLoss):
            if model.logspace:
                loss, loss_2d, loss_1d, clean1d, target1d = \
                    criterion(clean_pred.log1p(), t, a, b, return_parts=True)
            else:
                loss, loss_2d, loss_1d, clean1d, target1d = \
                    criterion(clean_pred, t, a, b, return_parts=True)
            total_loss += loss.item()
            if isinstance(loss_1d, torch.Tensor):
                total_loss_1d += loss_1d.item()
            if isinstance(loss_2d, torch.Tensor):
                total_loss_2d += loss_2d.item()
        elif criterion is not None:
            if model.logspace:
                loss = criterion(clean_pred.log1p(), y.log1p())
            else:
                loss = criterion(clean_pred, y)
            total_loss += loss.item()


        # Compute PSNR and entropy per batch
        if y is not None:
            batch_psnr = psnr(clean_pred, y)
            total_psnr += batch_psnr.sum().item()

        batch_entropy, batch_entropy_delta = calculate_batch_entropy(x, clean_pred)
        total_entropy += batch_entropy.sum()
        total_entropy_delta += batch_entropy_delta.sum()

        # Save some examples if requested
        if return_every > 0 and total_images % return_every == 0:
            # use .clone() to avoid memory leaks (pinned memory might cause this)
            examples.append((
                x[:4].detach().cpu().clone(),
                clean_pred[:4].detach().cpu().clone(),
                y[:4].detach().cpu().clone(),
                clean1d.detach().cpu().clone(),
                target1d.detach().cpu().clone()
            ))

        total_images += x.shape[0]

    results = {}
    img_pixels = x.shape[-2] * x.shape[-1]
    if y is not None:
        results["avg_psnr"] = total_psnr / total_images 
    if isinstance(criterion, CombinedLoss):
        if criterion.loss_1d is not None:
            results[f"avg_1d_{criterion.loss_1d.__class__.__name__}"] = total_loss_1d / total_images 
        if criterion.loss_2d is not None:
            results[f"avg_2d_{criterion.loss_2d.__class__.__name__}"] = total_loss_2d / total_images
        if criterion.loss_1d is not None and criterion.loss_2d is not None:
            results["avg_loss"] = (
                a * (total_loss_2d / img_pixels) + 
                b * (total_loss_1d / target1d.shape[-1])
            ) / total_images
    elif criterion is not None:
        results["avg_loss"] = (total_loss / total_images) / img_pixels

    results["output_entropy"] = total_entropy / total_images
    results["entropy_delta"] = total_entropy_delta / total_images

    return results, examples