import torch
from loss import prepare_profiles

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
def evaluate(model, loader, device, criterion=None, return_every=0):
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
    n_batches = 0
    examples = []

    for x, t in loader:
        if isinstance(t, torch.Tensor):
            y = t
        else:
            y, p = t
            p = (a.detach().cpu() for a in p)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        clean_pred = model(x)

        if criterion is not None:
            if isinstance(t, torch.Tensor):
                # Use t on correct device
                loss = criterion(clean_pred, y)
            else:
                loss = criterion(clean_pred, t)
            total_loss += loss.item()

        # Compute PSNR per batch
        batch_psnr = psnr(clean_pred, y)
        total_psnr += batch_psnr

        # Save some examples if requested
        if return_every > 0 and n_batches % return_every == 0:
            examples.append((
                x[:5].detach().cpu(),
                clean_pred[:5].detach().cpu(),
                t.detach().cpu()[:5] if isinstance(t, torch.Tensor) else (y.detach().cpu()[:5], p)
            ))

        n_batches += 1

    avg_loss = total_loss / n_batches if criterion is not None else None
    avg_psnr = total_psnr / n_batches

    return avg_loss, avg_psnr, examples

@torch.no_grad()
def evaluate_profile1d(model, loader, device, criterion=None, return_every=0):
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
        avg_mae: average MAE
        examples: list of tuples (input, clean_pred, target) for inspection
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    examples = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        clean_pred = model(x)

        if criterion is not None:
            loss = criterion(clean_pred, y)
            total_loss += loss.item()

        # Compute MAE per batch
        x_profile, y_profile = prepare_profiles(clean_pred, y)
        batch_mae = torch.nn.functional.l1_loss(x_profile, y_profile)
        total_mae += batch_mae


        # Save some examples if requested
        if return_every > 0 and n_batches % return_every == 0:
            examples.append((
                x.cpu(),
                clean_pred.cpu(),
                y
            ))

        n_batches += 1

    avg_loss = total_loss / n_batches if criterion is not None else None
    avg_mae = total_mae / n_batches

    return avg_loss, avg_mae, examples