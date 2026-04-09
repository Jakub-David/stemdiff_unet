from unet import ResidualUNet
from data import STEMDataset
from eval import evaluate
from plot import show_diffractograms, show_1D_profiles
from pavlina import PavlinaModel
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import datetime
import numpy as np
from tqdm import tqdm


# -------------------------------
# 1. Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- HELPER FUNCTION FOR NAMING ---
def generate_experiment_name(config):
    """
    Converts a config dictionary into a string: 
    'lr=0.001-base_channels=8-logspace=True'
    """
    # We sort the keys so that the same parameters always produce the same string
    parts = []
    for k in sorted(config.keys()):
        parts.append(f"{k}={config[k]}")
    return "-".join(parts)


def init_data(dataset_dir, batch_size, seed=seed):
    # Dataset & DataLoader
    dataset = STEMDataset(dataset_dir)

    # Define split fractions
    train_frac = 0.7
    val_frac = 0.15
    total_len = len(dataset)

    train_len = int(train_frac * total_len)
    val_len = int(val_frac * total_len)
    test_len = total_len - train_len - val_len  # make sure all samples are included
    assert total_len == test_len + val_len + train_len

    # Split dataset
    train_dataset, val_dataset, test_dataset =  random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)  # ensures reproducibility
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # shuffle training
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # no need to shuffle
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader

def train(config):
    # 1. Extract parameters from config
    dataset_dir = config['dataset_dir']
    lr = config['lr']
    num_epochs = config['num_epochs']
    log_interval = config.get('log_interval', 20)
    batch_size = config.get('batch_size', 32)
    
    # Model params
    model_params = config['model_params']

    train_loader, val_loader, test_loader = init_data(dataset_dir, batch_size)

    # 2. Generate unique experiment name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_string = generate_experiment_name(config)
    experiment_id = f"{timestamp}_{param_string}"

    checkpoint_dir = f"runs/{experiment_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model
    model = ResidualUNet(**model_params).to(device)

    # Loss & optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TensorBoard
    writer = SummaryWriter(f"runs/{experiment_id}")
    inputs_targets_logged = False

    print(f"Starting experiment: {experiment_id}")

    # -------------------------------
    # 2. Training loop
    # -------------------------------
    global_step = 0
    val_avg_loss = 0
    val_avg_psnr = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0

        for x, y in tqdm(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            clean_pred, background_pred = model(x)

            loss = criterion(clean_pred, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Logging
            if global_step % log_interval == 0:
                val_avg_loss, val_avg_psnr, examples = evaluate(
                    model, val_loader, device, criterion, return_examples=1
                )
                
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("AvgLoss/val", val_avg_loss, global_step)
                writer.add_scalar("AvgPSNR/val", val_avg_psnr, global_step)

                # Log Images to TensorBoard
                if examples is not None and len(examples) > 0:
                    x_batch, clean_batch, y_batch = examples[0]
                    log_idx = [0, 2, 6, 9, 14, 19, 20, 23, 26]
                    
                    if not inputs_targets_logged:
                        # We log a small subset (e.g., first 4) to save space
                        writer.add_images("Static/Input", x_batch[log_idx].detach().cpu(), global_step)
                        writer.add_images("Static/Target", y_batch[log_idx].detach().cpu(), global_step)
                        inputs_targets_logged = True

                    writer.add_images("Progress/Prediction", clean_batch[log_idx].detach().cpu(), global_step)
                
            global_step += 1

        # Log end of epoch
        avg_loss = epoch_loss / len(train_loader)
        val_avg_loss, val_avg_psnr, examples = evaluate(
                    model, val_loader, device, criterion, return_examples=1
        )
        
        writer.add_scalar("AvgLoss/train", avg_loss, global_step)
        writer.add_scalar("AvgLoss/val", val_avg_loss, global_step)
        writer.add_scalar("AvgPSNR/val", val_avg_psnr, global_step)

        # Log Images to TensorBoard
        if examples is not None and len(examples) > 0:
            x_batch, clean_batch, y_batch = examples[0]
            log_idx = [0, 2, 6, 9, 14, 19, 20, 23, 26]
            writer.add_images("EndOfEpoch/Prediction", clean_batch[log_idx].detach().cpu(), global_step)

        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val loss: {val_avg_loss:.6f}, Avg val psnr: {val_avg_psnr:.6f},")

        # -------------------------------
        # 3. Checkpointing
        # -------------------------------
        checkpoint_path = os.path.join(checkpoint_dir, f"residual_unet_epoch{epoch+1}.pt")
        model.save(checkpoint_path, epoch, optimizer, avg_loss)

    writer.close()

    return experiment_id

def test(experiment_id, batch_size=32, epoch=20, show_plots=True, show_all=True,
         show_idxs=None):
    # Reconstruct the path using the returned ID
    model_path = f"runs/{experiment_id}/residual_unet_epoch{epoch}.pt"


    train_loader, val_loader, test_loader = init_data(batch_size)

    # Model
    model, _ = ResidualUNet.load(model_path)
    model = model.to(device)

    avg_loss, avg_psnr, examples = evaluate(model, val_loader, device, return_examples=1)
    print(f"Test Results for {experiment_id}:")
    print("avg loss:", avg_loss)
    print("avg psnr:", avg_psnr)

    if not show_plots:
        return

    x, clean, y = examples[0]

    pav = PavlinaModel()
    pavlina_clean = pav(x)[0]
    idxs = range(batch_size) if show_all else [0, 2, 6, 9, 14, 19, 20, 23, 26]
    if show_idxs != None:
        idxs = show_idxs
    for i in idxs:
        show_diffractograms({
            "Original": x[i, 0], 
            "Result": clean[i, 0], 
            "Pavlina": pavlina_clean[i, :, :, 0]
        })
        show_1D_profiles({
            "Original": (x[i, 0].numpy(), "blue"), 
            "Target": (y[i, 0].numpy(), "--r"), 
            "Result": (clean[i, 0].numpy(), "green"), 
            "Pavlina": (pavlina_clean[i], "-.m")
        })


if __name__ == "__main__":
    config = {
        "dataset_dir": "dataset",
        "lr": 1e-4,
        "num_epochs": 20,
        "log_interval": 200,
        "batch_size": 32,
        "model_params": {
            "in_channels": 1,
            "base_channels": 8,
            "logspace": False,
            "normalize": False,
            "predict_background": True
        }
    }

    # Run training
    exp_id = train(config)
    
    # Run testing
    test(exp_id)