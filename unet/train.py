from model import ResidualUNet
from loss import CombinedLoss, RadialDistribution
from data import *
from eval import evaluate
from plot import create_profile_img
from torch.utils.data import DataLoader
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

def init_dataset(dataset_dir, batch_size, scale_factor, include_targets=True,
                 include_profiles = False, shuffle_val=False, same_key_batch=False):
    train_dataset = STEMDataset(
        dataset_dir,
        "train.h5",
        "train_target.h5" if include_targets else None,
        scale_factor,
        include_profiles
    )
    val_dataset = STEMDataset(
        dataset_dir,
        "val.h5",
        "val_target.h5" if include_targets else None,
        scale_factor,
        include_profiles
    )

    if same_key_batch:
        train_sampler = SameKeyBatchSampler(train_dataset.index_map, batch_size, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=train_sampler
    )

    if same_key_batch:
        val_sampler = SameKeyBatchSampler(val_dataset.index_map, batch_size, shuffle=shuffle_val)
    else:
        val_sampler = None
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=val_sampler
    )

    return train_loader, val_loader

def log_images(writer, dataset_type, global_step, examples, log_static, epoch_str, 
               individual_profiles, profile_scale):
    for i in range(len(examples)):
        x, clean, y = examples[i]
        if dataset_type == "preprocessed+profile":
            p = y[1:]
            y = y[0]
        else:
            p = y

        rad_dist = RadialDistribution(256 * profile_scale, 256 * profile_scale, x.device)
        
        # log first four images in a batch
        clean_log = clean[:4]
        writer.add_images(f"{epoch_str}Predictions/Prediction{i}", clean_log, global_step)

        if log_static:
            x_log = x[:4] 
            writer.add_images(f"StaticInputs/Input{i}", x_log, global_step)
            if dataset_type != "profile":
                y_log = y[:4]
                writer.add_images(f"StaticTargets/Target{i}", y_log, global_step)

        if "profile" in dataset_type:
            if individual_profiles:
                clean = clean[0][None] # add batch dim back
                p = [z[0][None] for z in p]
            writer.add_image(
                f"{epoch_str}Profiles/Profile{i}", 
                create_profile_img(clean, p, individual_profiles, rad_dist, profile_scale), 
                global_step
            )


def train(config: dict, experiment_name=None):
    # -------------------------------
    # Extract parameters from config
    # -------------------------------
    dataset_dir = config['dataset_dir']
    dataset_type = config['dataset_type']
    same_dataset_batch = config['same_dataset_batch']
    scale_factor = config['scale_factor']
    profile_scale = config['profile_scale']
    lr = config['lr']
    min_lr = config["min_lr"]
    num_epochs = config['num_epochs']
    log_interval = config.get('log_interval', 20)
    batch_size = config.get('batch_size', 32)
    
    model_params = config['model_params']

    # -------------------------------
    # Initialize dataset
    # -------------------------------
    if dataset_type == "preprocessed":
        train_loader, val_loader = init_dataset(dataset_dir, batch_size, scale_factor)
    elif dataset_type == "profile":
        train_loader, val_loader = init_dataset(
            dataset_dir, 
            batch_size, 
            scale_factor,
            include_targets=False,
            include_profiles=True,
            same_key_batch=same_dataset_batch
        )
    elif dataset_type == "preprocessed+profile":
        train_loader, val_loader = init_dataset(
            dataset_dir, 
            batch_size, 
            scale_factor,
            include_targets=True,
            include_profiles=True,
            same_key_batch=same_dataset_batch
        )

    # -------------------------------
    # Prepare experiment directory
    # -------------------------------
    experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name != None:
        experiment_id += "_" + experiment_name

    checkpoint_dir = f"runs2/{experiment_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    # -------------------------------
    # Model
    # -------------------------------
    ckpt = config.get("ckpt")
    ckpt_epoch = config.get("ckpt_epoch", "")
    if ckpt != None:
        model, ckpt_config = ResidualUNet.load("runs2/", f"{ckpt}*/*{ckpt_epoch}.pt")
        model = model.to(device)
    else:
        model = ResidualUNet(**model_params).to(device)

    # -------------------------------
    # Loss & optimizer
    # -------------------------------
    if dataset_type == "profile":
        criterion = CombinedLoss(device, model.logspace, profile_scale, not same_dataset_batch, include_2d=False)
    elif dataset_type == "preprocessed+profile":
        criterion = CombinedLoss(device, model.logspace, profile_scale, not same_dataset_batch, include_2d=True)
    else:
        criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=min_lr
    )

    # Loss weights
    a, b = 1, 1

    # -------------------------------
    # TensorBoard
    # -------------------------------
    writer = SummaryWriter(f"runs2/{experiment_id}")
    inputs_targets_logged = False

    print(f"Starting experiment: {experiment_id}")

    # -------------------------------
    # Training loop
    # -------------------------------
    global_step = 0
    for epoch in (pbar := tqdm(range(num_epochs))):
        model.train()
        epoch_loss = 0.0

        for x, y in tqdm(train_loader):
            x = x.to(device, non_blocking=True)
            if isinstance(y, torch.Tensor):
                y = y.to(device, non_blocking=True)
            else:
                y = [z.to(device, non_blocking=True) for z in y]

            optimizer.zero_grad()

            clean_pred = model(x)

            if dataset_type == "preprocessed+profile":
                a = float(np.interp(epoch, [0, num_epochs], [1, 0.7]))
                b = 1 - a
                loss = criterion(clean_pred, y, a, b)
            elif dataset_type == "profile":
                loss = criterion(x, y)
            else:
                if model.logspace:
                    loss = criterion(clean_pred, torch.log1p(y))
                else:
                    loss = criterion(clean_pred, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if global_step % 50 == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

            # -------------------------------
            # Logging during epoch
            # -------------------------------
            writer.add_scalar("Train/loss", loss.item(), global_step)
            if log_interval > 0 and global_step % log_interval == 0:
                eval_metrics, examples = evaluate(
                    model, val_loader, device, criterion, return_every=5, a=a, b=b
                )

                for n, v in eval_metrics.items():
                    writer.add_scalar(f"DuringEpoch/val_{n}", v, global_step)
                
                # -------------------------------
                # Log Images to TensorBoard
                # -------------------------------
                log_images(
                    writer, 
                    dataset_type, 
                    global_step, 
                    examples, 
                    not inputs_targets_logged, 
                    "DuringEpoch",
                    not same_dataset_batch,
                    profile_scale
                )
                inputs_targets_logged = True
                
                # Free memory
                del examples
                
            global_step += 1

        # -------------------------------
        # Log end of epoch
        # -------------------------------
        # Avg. loss
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Train/avg_loss", avg_loss, epoch)

        # Metrics
        eval_metrics, examples = evaluate(
            model, val_loader, device, criterion, return_every=5, a=a, b=b
        )
        
        for n, v in eval_metrics.items():
            writer.add_scalar(f"EndOfEpoch/val_{n}", v, epoch)

        # Log Images to TensorBoard
        
        log_images(
            writer, 
            dataset_type, 
            global_step, 
            examples, 
            not inputs_targets_logged, 
            "EndOfEpoch",
            not same_dataset_batch,
            profile_scale
        )
        inputs_targets_logged = True
        
        # Free memory
        del examples

        # Write progress to console             
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val loss: {eval_metrics["avg_loss"]:.6f}, Avg val psnr: {eval_metrics["avg_psnr"]:.6f}")

        # Log loss weights
        if dataset_type == "preprocessed+profile":
            writer.add_scalar("Hyperparameters/WeightA", a, epoch)
            writer.add_scalar("Hyperparameters/WeightB", b, epoch)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Hyperparameters/LearningRate", current_lr, epoch)

        # -------------------------------
        # Learning rate step
        # -------------------------------
        scheduler.step()

        # -------------------------------
        # Checkpointing
        # -------------------------------
        checkpoint_path = os.path.join(checkpoint_dir, f"residual_unet_epoch{epoch+1}.pt")
        model.save(checkpoint_path, epoch, optimizer, avg_loss)

    writer.close()

    return experiment_id

if __name__ == "__main__":
    config = {
        "dataset_dir": "dataset_filtered",
        # Possible dataset_type values: "preprocessed", "profile", "preprocessed+profile"
        "dataset_type": "preprocessed+profile",
        # Does nothing for "preprocessed"
        "same_dataset_batch": False,
        "scale_factor": 1,
        "profile_scale": 1,
        "lr": 3e-3,
        "min_lr": 3e-5,
        "num_epochs": 60,
        "log_interval": -1,
        "batch_size": 32,
        "model_params": {
            "in_channels": 1,
            "base_channels": 2,
            "normalize": False,
            "logspace": True,
            "predict_background": True
        },
    }

    exp_id = train(config, "combined_bc2_logspace_end0.7_individual")
    
