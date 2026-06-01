from model import ResidualUNet
from loss import profile_1d_loss, combined_loss
from data import *
from eval import evaluate, evaluate_profile1d
from plot import create_profile_img
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
# Setup
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

def init_old_data(dataset_dir, batch_size, seed=seed):
    # Dataset & DataLoader
    dataset = PreprocessedDataset(
        os.path.join(dataset_dir, "x_input.h5"),
        os.path.join(dataset_dir, "x_target.h5")
    )

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

def init_preprocessed(dataset_dir, batch_size, scale_factor=None):
    train_dataset = PreprocessedDataset(
        os.path.join(dataset_dir, "train.h5"),
        os.path.join(dataset_dir, "train_target.h5"),
        scale_factor
    )
    val_dataset = PreprocessedDataset(
        os.path.join(dataset_dir, "val.h5"),
        os.path.join(dataset_dir, "val_target.h5"),
        scale_factor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader


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

def init_profile1d(dataset_dir, batch_size, scale_factor, include_targets=False, shuffle_val=False):
    train_dataset = Profile1DDataset(
        os.path.join(dataset_dir, "train.h5"),
        os.path.join(dataset_dir),
        scale_factor,
        os.path.join(dataset_dir, "train_target.h5") if include_targets else None
    )
    val_dataset = Profile1DDataset(
        os.path.join(dataset_dir, "val.h5"),
        os.path.join(dataset_dir),
        scale_factor,
        os.path.join(dataset_dir, "val_target.h5") if include_targets else None
    )

    train_sampler = SameKeyBatchSampler(train_dataset.index_map, batch_size, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=train_sampler
    )

    val_sampler = SameKeyBatchSampler(val_dataset.index_map, batch_size, shuffle=shuffle_val)
    val_loader = DataLoader(
        val_dataset,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=val_sampler
    )

    return train_loader, val_loader

def log_images(writer, dataset_type, global_step, examples, log_static, epoch_str):
    for i in range(len(examples)):
        x, clean, y = examples[i]
        if dataset_type == "preprocessed+profile":
            y, p = y
        else:
            p = y
        
        # log first four images in a batch
        clean_log = clean[[0, 1, 2, 3]]
        writer.add_images(f"{epoch_str}/Prediction{i}", clean_log, global_step)

        if log_static:
            x_log = x[[0, 1, 2, 3]] 
            writer.add_images(f"Static/Input{i}", x_log, global_step)
            if dataset_type != "profile":
                y_log = y[[0, 1, 2, 3]]
                writer.add_images(f"Static/Target{i}", y_log, global_step)

        if "profile" in dataset_type:
            writer.add_image(f"{epoch_str}/Profile{i}", create_profile_img(clean.detach().cpu(), p), global_step)


def train(config: dict, experiment_name=None):
    # -------------------------------
    # Extract parameters from config
    # -------------------------------
    dataset_dir = config['dataset_dir']
    dataset_type = config['dataset_type']
    scale_factor = config['scale_factor']
    lr = config['lr']
    min_lr = config["min_lr"]
    num_epochs = config['num_epochs']
    log_interval = config.get('log_interval', 20)
    batch_size = config.get('batch_size', 32)
    
    model_params = config['model_params']

    # -------------------------------
    # Initialize dataset
    # -------------------------------
    if dataset_type == "old":
        train_loader, val_loader, test_loader = init_old_data(dataset_dir, batch_size)
    elif dataset_type == "preprocessed":
        train_loader, val_loader = init_preprocessed(dataset_dir, batch_size, scale_factor)
    elif dataset_type == "augmented":
        train_loader, val_loader = init_augmented(dataset_dir, batch_size)
    elif dataset_type == "resized_augmented":
        train_loader, val_loader = init_augmented(dataset_dir, batch_size, resized=True)
    elif dataset_type == "profile":
        train_loader, val_loader = init_profile1d(dataset_dir, batch_size, scale_factor)
    elif dataset_type == "preprocessed+profile":
        train_loader, val_loader = init_profile1d(dataset_dir, batch_size, scale_factor, include_targets=True)

    # -------------------------------
    # Prepare experiment directory
    # -------------------------------
    experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name != None:
        experiment_id += "_" + experiment_name

    checkpoint_dir = f"runs/{experiment_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    # -------------------------------
    # Model
    # -------------------------------
    ckpt = config.get("ckpt")
    ckpt_epoch = config.get("ckpt_epoch", "")
    if ckpt != None:
        model, ckpt_config = ResidualUNet.load("runs/", f"{ckpt}*/*{ckpt_epoch}.pt")
        model = model.to(device)
    else:
        model = ResidualUNet(**model_params).to(device)

    # -------------------------------
    # Loss & optimizer
    # -------------------------------
    if dataset_type == "profile":
        criterion = profile_1d_loss
    elif dataset_type == "preprocessed+profile":
        criterion = combined_loss
    else:
        criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=min_lr
    )

    # -------------------------------
    # TensorBoard
    # -------------------------------
    writer = SummaryWriter(f"runs/{experiment_id}")
    inputs_targets_logged = False

    print(f"Starting experiment: {experiment_id}")

    # -------------------------------
    # Training loop
    # -------------------------------
    global_step = 0
    val_avg_loss = 0
    val_avg_psnr = 0
    for epoch in (pbar := tqdm(range(num_epochs))):
        model.train()
        epoch_loss = 0.0

        for x, y in tqdm(train_loader):
            x = x.to(device, non_blocking=True)
            if isinstance(y, torch.Tensor):
                y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            clean_pred = model(x)

            if dataset_type == "preprocessed+profile":
                # TODO: add logspace support
                a = float(np.interp([epoch], [0, num_epochs], [1, 0.1])[0])
                b = 1 - a
                loss = criterion(clean_pred, y, a, b)
            else:
                if model_params["logspace"]:
                    loss = criterion(clean_pred, torch.log1p(y))
                else:
                    loss = criterion(clean_pred, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if global_step % 10 == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

            # -------------------------------
            # Logging during epoch
            # -------------------------------
            if log_interval > 0 and global_step % log_interval == 0:
                if dataset_type == "profile":
                    val_avg_loss, val_avg_mae, examples = evaluate_profile1d(
                        model, val_loader, device, criterion, return_every=5
                    )
                else:
                    val_avg_loss, val_avg_psnr, examples = evaluate(
                        model, val_loader, device, criterion, return_every=5
                    )

                
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("AvgLoss/val", val_avg_loss, global_step)
                if dataset_type == "profile":
                    writer.add_scalar("AvgMAE/val", val_avg_mae, global_step)
                else:
                    writer.add_scalar("AvgPSNR/val", val_avg_psnr, global_step) 

                # -------------------------------
                # Log Images to TensorBoard
                # -------------------------------
                if examples is not None and len(examples) > 0:
                    log_images(writer, dataset_type, global_step, examples, not inputs_targets_logged, "DuringEpoch")
                    inputs_targets_logged = True
                
                # Free memory
                del examples
                
            global_step += 1

        # -------------------------------
        # Log end of epoch
        # -------------------------------
        avg_loss = epoch_loss / len(train_loader)
        if dataset_type == "profile":
            val_avg_loss, val_avg_mae, examples = evaluate_profile1d(
                model, val_loader, device, criterion, return_every=5
            )
        else:
            val_avg_loss, val_avg_psnr, examples = evaluate(
                model, val_loader, device, criterion, return_every=5
            )
        
        writer.add_scalar("AvgLoss/train", avg_loss, global_step)
        writer.add_scalar("AvgLoss/val", val_avg_loss, global_step)
        if dataset_type == "profile":
            writer.add_scalar("AvgMAE/val", val_avg_mae, global_step)
        else:
            writer.add_scalar("AvgPSNR/val", val_avg_psnr, global_step) 

        # -------------------------------
        # Log Images to TensorBoard
        # -------------------------------
        if examples is not None and len(examples) > 0:
            log_images(writer, dataset_type, epoch, examples, not inputs_targets_logged, "EndOfEpoch")
            inputs_targets_logged = True
        
        # Free memory
        del examples
                    
        if dataset_type == "profile":
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val loss: {val_avg_loss:.6f}, Avg val mea: {val_avg_mae:.6f},")
        else:
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val loss: {val_avg_loss:.6f}, Avg val psnr: {val_avg_psnr:.6f},")

        # -------------------------------
        # Learning rate step
        # -------------------------------
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Hyperparameters/LearningRate", current_lr, epoch)

        # Log loss weights

        if dataset_type == "preprocessed+profile":
            writer.add_scalar("Hyperparameters/WeightA", a, epoch)

        # -------------------------------
        # Checkpointing
        # -------------------------------
        checkpoint_path = os.path.join(checkpoint_dir, f"residual_unet_epoch{epoch+1}.pt")
        model.save(checkpoint_path, epoch, optimizer, avg_loss)

    writer.close()

    return experiment_id

if __name__ == "__main__":
    # config = {
    #     "dataset_dir": "dataset1.1",
    #     # Possible dataset_type values: "preprocessed", "profile"
    #     "dataset_type": "preprocessed",
    #     "scale_factor": 1,
    #     "lr": 3e-2,
    #     "min_lr": 3e-5,
    #     "num_epochs": 200,
    #     "log_interval": -1,
    #     "batch_size": 32,
    #     "model_params": {
    #         "in_channels": 1,
    #         "base_channels": 1,
    #         "normalize": True,
    #         "logspace": True,
    #         "predict_background": True
    #     },
    #     # "ckpt": "20260417_154943",
    #     # "ckpt": "20260519_193326_preprocessed_gaussian_4x",
    #     # "ckpt_epoch": 40
    # }

    config = {
        "dataset_dir": "dataset1.1",
        # Possible dataset_type values: "preprocessed", "profile", "preprocessed+profile"
        "dataset_type": "preprocessed",
        "scale_factor": 1,
        "lr": 1e-3,
        "min_lr": 6e-5,
        "num_epochs": 100,
        "log_interval": -1,
        "batch_size": 32,
        "model_params": {
            "in_channels": 1,
            "base_channels": 1,
            "normalize": False,
            "logspace": True,
            "predict_background": True
        },
        # "ckpt": "20260417_154943",
        # "ckpt": "20260520_163333_preprocessed_gaussian_2x",
        # "ckpt_epoch": 40
    }

    # Run training
    # exp_id = train(config, "preprocessed_gaussian_1x_bchannels1_logloss_logspace")
    # exp_id = train(config, "profile_2x_gaussian_v2")
    # exp_id = train(config, "combined_g2x_precalc_cal_const")
    # exp_id = train(config, "combined_g1x_bchannels2")
    exp_id = train(config, "preprocessed_gaussian_1x_bc1_global_skip_improved_block")
    
