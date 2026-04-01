from unet import ResidualUNet
from data import STEMDataset
from eval import evaluate
from plot import show_diffractograms, show_1D_profiles
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

# Paths
input_dir = "dataset/x_input"
target_dir = "dataset/x_target"

# Hyperparameters
batch_size = 32
lr = 1e-3
num_epochs = 20
patch_size = None
log_interval = 20  # log every N batches

def init_data():
    # Dataset & DataLoader
    dataset = STEMDataset(input_dir, target_dir)

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

def main(experiment_name):
    train_loader, val_loader, test_loader = init_data()

    checkpoint_dir = f"checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model
    model = ResidualUNet(in_channels=1, base_channels=8, logspace=True).to(device)

    # Loss & optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TensorBoard
    writer = SummaryWriter(f"runs/{experiment_name}")

    # -------------------------------
    # 2. Training loop
    # -------------------------------
    global_step = 0
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
                val_avg_loss, val_avg_psnr, _ = evaluate(model, val_loader, device, criterion)
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("AvgLoss/val", val_avg_loss, global_step)
                writer.add_scalar("AvgPSNR/val", val_avg_psnr, global_step)
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val loss: {val_avg_loss:.6f}, Avg val psnr: {val_avg_psnr:.6f},")

        # -------------------------------
        # 3. Checkpointing
        # -------------------------------
        checkpoint_path = os.path.join(checkpoint_dir, f"residual_unet_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)

    writer.close()

def test(model_path):
    train_loader, val_loader, test_loader = init_data()

    # Model
    model = ResidualUNet(in_channels=1, base_channels=8, logspace=True).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    avg_loss, avg_psnr, examples = evaluate(model, test_loader, device, return_examples=1)
    print("avg loss:", avg_loss)
    print("avg psnr:", avg_psnr)
    x, clean, y = examples[0]
    for i in range(20):
        show_diffractograms(x[i, 0], clean[i, 0])
        show_1D_profiles(x[i, 0].numpy(), y[i, 0].numpy())
        show_1D_profiles(x[i, 0].numpy(), clean[i, 0].numpy())


if __name__ == "__main__":
    experiment_name = "logspace+normalize"
    experiment_dir = f"{datetime.datetime.now()}_{experiment_name}"
    main(experiment_dir)
    test(f"checkpoints/{experiment_dir}/residual_unet_epoch20.pt")