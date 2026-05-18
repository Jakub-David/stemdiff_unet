from model import ResidualUNet
from loss import profile_1d_loss
from data import *
from eval import evaluate, evaluate_profile1d
from plot import show_diffractograms, show_1D_profiles, create_profile_img
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import random
import os
import datetime
import numpy as np
from tqdm import tqdm
import json


# -------------------------------
# 1. Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

def init_augmented(dataset_dir, batch_size, shuffle_val=False, resized=False):
    if resized:
        train_dataset = ResizedAugmentedDataset(os.path.join(dataset_dir, "train.h5"))
        val_dataset = ResizedAugmentedDataset(os.path.join(dataset_dir, "val.h5"))
    else:
        train_dataset = AugmentedDataset(os.path.join(dataset_dir, "train.h5"))
        val_dataset = AugmentedDataset(os.path.join(dataset_dir, "val.h5"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader

def init_profile1d(dataset_dir, batch_size, shuffle_val=False):
    train_dataset = Profile1DDataset(
        os.path.join(dataset_dir, "train.h5"),
        os.path.join(dataset_dir),
    )
    val_dataset = Profile1DDataset(
        os.path.join(dataset_dir, "val.h5"),
        os.path.join(dataset_dir),
    )

    train_sampler = SameKeyBatchSampler(train_dataset.index_map, batch_size, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=train_sampler
    )

    val_sampler = SameKeyBatchSampler(val_dataset.index_map, batch_size, shuffle=shuffle_val)
    val_loader = DataLoader(
        val_dataset,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=val_sampler
    )

    return train_loader, val_loader

def train(config: dict, experiment_name=None):
    # 1. Extract parameters from config
    dataset_dir = config['dataset_dir']
    dataset_type = config['dataset_type']
    lr = config['lr']
    min_lr = config["min_lr"]
    num_epochs = config['num_epochs']
    log_interval = config.get('log_interval', 20)
    batch_size = config.get('batch_size', 32)
    
    # Model params
    model_params = config['model_params']

    if dataset_type == "default":
        train_loader, val_loader, test_loader = init_data(dataset_dir, batch_size)
    elif dataset_type == "augmented":
        train_loader, val_loader = init_augmented(dataset_dir, batch_size)
    elif dataset_type == "resized_augmented":
        train_loader, val_loader = init_augmented(dataset_dir, batch_size, resized=True)
    elif dataset_type == "profile":
        train_loader, val_loader = init_profile1d(dataset_dir, batch_size)

    # 2. Generate unique experiment name
    experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name != None:
        experiment_id += "_" + experiment_name

    checkpoint_dir = f"runs/{experiment_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    # Model
    ckpt = config.get("ckpt")
    ckpt_epoch = config.get("ckpt_epoch", "")
    if ckpt != None:
        model, ckpt_config = ResidualUNet.load("runs/", f"{ckpt}*/*{ckpt_epoch}.pt")
        model = model.to(device)
    else:
        model = ResidualUNet(**model_params).to(device)

    # Loss & optimizer
    if dataset_type == "profile":
        criterion = profile_1d_loss
    else:
        criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=min_lr
    )

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
            if isinstance(y, torch.Tensor):
                y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            clean_pred = model(x)

            loss = criterion(clean_pred, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Logging
            if log_interval > 0 and global_step % log_interval == 0:
                if dataset_type == "profile":
                    val_avg_loss, val_avg_mae, examples = evaluate_profile1d(
                        model, val_loader, device, criterion, return_examples=10
                    )
                else:
                    val_avg_loss, val_avg_psnr, examples = evaluate(
                        model, val_loader, device, criterion, return_examples=1
                    )

                
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("AvgLoss/val", val_avg_loss, global_step)
                if dataset_type == "profile":
                    writer.add_scalar("AvgMAE/val", val_avg_mae, global_step)
                else:
                    writer.add_scalar("AvgPSNR/val", val_avg_psnr, global_step) 

                # Log Images to TensorBoard
                if examples is not None and len(examples) > 0:
                    x_batch, clean_batch, y_batch = examples[0]
                    # log_idx = [0, 2, 6, 9, 14, 19, 20, 23, 26]
                    log_idx = [0, 2, 6, 9, 14]
                    
                    if not inputs_targets_logged:
                        # We log a small subset (e.g., first 4) to save space
                        writer.add_images("Static/Input", np.log10(x_batch[log_idx].detach().cpu() + 1), global_step)
                        if dataset_type != "profile":
                            writer.add_images("Static/Target", y_batch[log_idx].detach().cpu(), global_step)
                        inputs_targets_logged = True

                    writer.add_images("Progress/Prediction", clean_batch[log_idx].detach().cpu(), global_step)
                    if dataset_type == "profile":
                        for i in range(len(examples)):
                            x_i, clean_i, y_i = examples[i]
                            writer.add_image(f"Progress/Profile{i}", create_profile_img(clean_i.detach().cpu(), y_i), global_step)
                
            global_step += 1

        # Log end of epoch
        avg_loss = epoch_loss / len(train_loader)
        if dataset_type == "profile":
            val_avg_loss, val_avg_mae, examples = evaluate_profile1d(
                model, val_loader, device, criterion, return_examples=10
            )
        else:
            val_avg_loss, val_avg_psnr, examples = evaluate(
                model, val_loader, device, criterion, return_examples=1
            )
        
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("AvgLoss/val", val_avg_loss, global_step)
        if dataset_type == "profile":
            writer.add_scalar("AvgMAE/val", val_avg_mae, global_step)
        else:
            writer.add_scalar("AvgPSNR/val", val_avg_psnr, global_step) 

        # Log Images to TensorBoard
        if examples is not None and len(examples) > 0:
            x_batch, clean_batch, y_batch = examples[0]
            # log_idx = [0, 2, 6, 9, 14, 19, 20, 23, 26]
            log_idx = [0, 2, 6, 9, 14]
            writer.add_images("EndOfEpoch/Prediction", clean_batch[log_idx].detach().cpu(), global_step)
            if dataset_type == "profile":
                for i in range(len(examples)):
                    x_i, clean_i, y_i = examples[i]
                    writer.add_image(f"EndOfEpoch/Profile{i}", create_profile_img(clean_i.detach().cpu(), y_i), global_step)

            if not inputs_targets_logged:
                # We log a small subset (e.g., first 4) to save space
                writer.add_images("Static/Input", np.log10(x_batch[log_idx].detach().cpu() + 1), global_step)
                if dataset_type != "profile":
                    writer.add_images("Static/Target", y_batch[log_idx].detach().cpu(), global_step)
                    
        if dataset_type == "profile":
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val loss: {val_avg_loss:.6f}, Avg val mea: {val_avg_mae:.6f},")
        else:
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val loss: {val_avg_loss:.6f}, Avg val psnr: {val_avg_psnr:.6f},")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Hyperparameters/LearningRate", current_lr, epoch)

        # -------------------------------
        # 3. Checkpointing
        # -------------------------------
        checkpoint_path = os.path.join(checkpoint_dir, f"residual_unet_epoch{epoch+1}.pt")
        model.save(checkpoint_path, epoch, optimizer, avg_loss)

    writer.close()

    return experiment_id

def test(experiment_id, dataset_dir, batch_size=32, epoch=20, show_plots=True, 
         show_all=True, show_idxs=None):
    # Reconstruct the path using the returned ID
    model_path = f"runs/{experiment_id}/residual_unet_epoch{epoch}.pt"


    # train_loader, val_loader, test_loader = init_data(dataset_dir, batch_size)
    train_loader, val_loader = init_augmented(dataset_dir, batch_size, True)

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

    idxs = range(batch_size) if show_all else [0, 2, 6, 9, 14, 19, 20, 23, 26]
    if show_idxs != None:
        idxs = show_idxs
    for i in idxs:
        show_diffractograms({
            "Original": x[i, 0], 
            "Result": clean[i, 0], 
            "Target": y[i, 0]
        })
        show_1D_profiles({
            "Original": (x[i, 0].numpy(), "blue"), 
            "Target": (y[i, 0].numpy(), "--r"), 
            "Result": (clean[i, 0].numpy(), "green"), 
        })


if __name__ == "__main__":
    config = {
        "dataset_dir": "dataset_filtered",
        "dataset_type": "profile",
        "lr": 1e-5,
        "min_lr": 1e-8,
        "num_epochs": 40,
        "log_interval": -1,
        "batch_size": 50,
        "model_params": {
            "in_channels": 1,
            "base_channels": 4,
            "logspace": False,
            "normalize": True,
            "predict_background": True
        },
        "ckpt": "20260417_154943",
        "ckpt_epoch": 40
    }

    # Run training
    exp_id = train(config, "all_profiles")
    
    # Run testing
    # test(exp_id, config["dataset_dir"])

    # test(
    #     "20260415_195938_batch_size=32-dataset_dir=dataset1.1-log_interval=-1-lr=0.001-min_lr=1e-06-model_params={'in_channels': 1, 'base_channels': 8, 'logspace': False, 'normalize': True, 'predict_background': True}-num_epochs=40", 
    #     config["dataset_dir"],
    #     32, 34)