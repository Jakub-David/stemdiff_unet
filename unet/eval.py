import torch
from loss import CombinedLoss

def psnr(pred, target, max_val=11810):
    """
    Compute Peak Signal-to-Noise Ratio
    pred, target: tensors [B, C, H, W]
    max_val: maximum possible pixel value
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

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
    n_batches = 0
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
                loss, loss_2d, loss_1d = criterion(clean_pred.log1p(), t, a, b, return_parts=True)
            else:
                loss, loss_2d, loss_1d = criterion(clean_pred, t, a, b, return_parts=True)
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


        # Compute PSNR per batch
        if y is not None:
            batch_psnr = psnr(clean_pred, y)
            total_psnr += batch_psnr

        # Save some examples if requested
        if return_every > 0 and n_batches % return_every == 0:
            # use .clone() to avoid memory leaks (pinned memory might cause this)
            examples.append((
                x.detach().cpu().clone(),
                clean_pred.detach().cpu().clone(),
                t.detach().cpu().clone() if isinstance(t, torch.Tensor) else [z.detach().cpu().clone() for z in t]
            ))

        n_batches += 1

    results = {}
    if criterion is not None:
        results["avg_loss"] = total_loss / n_batches
    if y is not None:
        results["avg_psnr"] = total_psnr / n_batches 
    if isinstance(criterion, CombinedLoss):
        if criterion.loss_1d is not None:
            results[f"avg_1d_{criterion.loss_1d.__class__.__name__}"] = total_loss_1d / n_batches 
        if criterion.loss_2d is not None:
            results[f"avg_2d_{criterion.loss_2d.__class__.__name__}"] = total_loss_2d / n_batches 

    return results, examples