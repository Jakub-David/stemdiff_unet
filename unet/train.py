from model import ResidualUNet
from loss import CombinedLoss
import loss
from data import *
from eval import evaluate
from plot import create_profile_img
from torch.utils.data import DataLoader
import torch
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
                 include_profiles = False, same_key_batch=False):
    train_dataset = STEMDataset(
        dataset_dir,
        "train.h5",
        "train_target.h5" if include_targets else None,
        scale_factor,
        include_profiles
    )
    val_dataset = STEMDataset(
        dataset_dir if dataset_dir != "dataset_all" else "dataset",
        "val.h5",
        "val_target.h5" if include_targets else None,
        scale_factor,
        True
    )

    if same_key_batch:
        train_sampler = SameKeyBatchSampler(train_dataset.index_map, batch_size, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if same_key_batch else batch_size,
        shuffle=None if same_key_batch else True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=train_sampler
    )

    val_sampler = SameKeyBatchSampler(val_dataset.index_map, batch_size)
    val_loader = DataLoader(
        val_dataset,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=val_sampler
    )

    return train_loader, val_loader

def log_images(writer, global_step, examples, log_static, epoch_str):
    for i in range(len(examples)):
        x, clean, y, clean1d, target1d = examples[i]
        
        # log first four images in a batch
        clean_log = clean[:4]
        writer.add_images(f"{epoch_str}Predictions/Prediction{i}", clean_log, global_step)

        if log_static:
            x_log = x[:4] 
            writer.add_images(f"StaticInputs/Input{i}", x_log, global_step)
            if y is not None:
                y_log = y[:4]
                writer.add_images(f"StaticTargets/Target{i}", y_log, global_step)

        writer.add_image(
            f"{epoch_str}Profiles/Profile{i}", 
            create_profile_img(clean1d, target1d), 
            global_step
        )

def serialize_config(cfg):
    serialized = cfg.copy()

    for key, obj in serialized.items():
        # Check if the item is a PyTorch Module (Losses are Modules)
        if isinstance(obj, torch.nn.Module):
            serialized[key] = {
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "params": {
                    k: v
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, dict))
                },
            }
    return serialized

def train(config: dict, experiment_name=None):
    """
    Example config is in __main__
    """
    # -------------------------------
    # Extract parameters from config
    # -------------------------------
    dataset_dir = config['dataset_dir']
    loss_2d = config['loss_2d']
    loss_1d = config['loss_1d']
    loss_2d_final_w = config['loss_2d_final_w']
    same_sample_batch = config['same_sample_batch']
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
    train_loader, val_loader = init_dataset(
        dataset_dir, 
        batch_size, 
        scale_factor,
        include_targets=loss_2d is not None,
        include_profiles=loss_1d is not None,
        same_key_batch=same_sample_batch
    )

    # -------------------------------
    # Prepare experiment directory
    # -------------------------------
    experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name != None:
        experiment_id += "_" + experiment_name

    checkpoint_dir = f"runs/{experiment_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(serialize_config(config), f, indent=1)

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
    criterion = CombinedLoss(
        device, 
        config["logspace"], 
        profile_scale, 
        individual_profiles=not same_sample_batch, 
        loss_1d=loss_1d, 
        loss_2d=loss_2d,
        l1_reg=config["l1_regularization"],
        total_variation=config["total_variation"],
        local_cons_reg=config["local_consistency_reg"],
        local_cons_noise_c=config["local_consistency_noise_constant"]
    )
    eval_criterion = CombinedLoss(
        device, 
        config["logspace"], 
        profile_scale, 
        individual_profiles=False, 
        loss_1d=torch.nn.L1Loss(reduction="sum"), 
        loss_2d=torch.nn.HuberLoss(reduction="sum") if loss_2d is not None else None
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=min_lr
    )

    # Loss weights
    interpolate_weights = loss_2d is not None and loss_1d is not None
    a, b = 1, 1

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
    for epoch in (pbar := tqdm(range(num_epochs))):
        model.train()
        epoch_loss = 0.0

        for data_dict in tqdm(train_loader):
            data_dict = {n: t.to(device, non_blocking=True) for n, t in data_dict.items()}

            optimizer.zero_grad()

            clean_pred = model(data_dict["original_image"])

            if interpolate_weights:
                a = float(np.interp(epoch, [0, num_epochs], [1, loss_2d_final_w]))
                b = 1 - a
            loss = criterion(clean_pred, data_dict, a, b)

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
                    model, val_loader, device, eval_criterion, return_every=10, a=a, b=b
                )

                for n, v in eval_metrics.items():
                    if v is not None:
                        writer.add_scalar(f"DuringEpoch/val_{n}", v, global_step)
                
                # -------------------------------
                # Log Images to TensorBoard
                # -------------------------------
                log_images(
                    writer, 
                    global_step, 
                    examples, 
                    not inputs_targets_logged, 
                    "DuringEpoch"
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
            model, val_loader, device, eval_criterion, return_every=10, a=a, b=b
        )
        
        for n, v in eval_metrics.items():
            if v is not None:
                writer.add_scalar(f"EndOfEpoch/val_{n}", v, epoch)

        # Log Images to TensorBoard
        
        log_images(
            writer,  
            epoch, 
            examples, 
            not inputs_targets_logged, 
            "EndOfEpoch"
        )
        inputs_targets_logged = True
        
        # Free memory
        del examples

        # Write progress to console             
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg train Loss: {avg_loss:.6f}, Avg val reverse kl div: {eval_metrics["reverse_kl_div"]:.6f}")

        # Log loss weights
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
        # Directory containing the dataset
        "dataset_dir": "dataset",
        # 2D loss, can be None
        "loss_2d": None, #loss.SymmetricMAPELoss(),
        # 1D loss, can be None
        "loss_1d": None, #loss.SymmetricMAPELoss(), #torch.nn.HuberLoss(),
        # Final weight of 2d loss
        # (Used only if both losses are used)
        "loss_2d_final_w": 0.7,
        # Apply sparsity regularization on the network output
        # This is the weight for the regularization, 0 means off
        "l1_regularization": 0.5,
        # Total variation reg
        "total_variation": 0,
        # Local consistency loss weight
        "local_consistency_reg": 0.5,
        # This constant controls noise level, higher value means more noise reduction
        # It is a multiplier for noise level estimated for each image
        "local_consistency_noise_constant": 0.3,
        # 2d loss, local consistency (final sparse error) and l1 is calculated on log(x + 1) inputs
        "logspace": True,
        # Batches contain images only for one sample (e.g. a batch contains only Au)
        "same_sample_batch": False,
        # Rescale input images and 2D targets
        "scale_factor": 1,
        # Scale for 1d targets and rescale nn outputs for 1d profile calculation
        # For "2D" dataset only used in logging
        "profile_scale": 1,
        # Initial learning rate
        "lr": 1e-3,
        # Final learning rate (cosine decay)
        "min_lr": 1e-5,
        # Number of training epochs
        "num_epochs": 20,
        # Log every n steps, n = -1 no logging
        # Does not affect loss logging and lagging at the end of epoch
        "log_interval": -1,
        # Batch size
        "batch_size": 32,
        # Parameters for model
        "model_params": {
            # Number of channels of input data, should be 1
            "in_channels": 1,
            # Number of channels on the first level of unet
            "base_channels": 4,
            # If true, Level n has `base_channels + (n - 1)` channels;
            # otherwise, level n has `base_channels * 2^n` channels
            "reduced_channels": True,
            # Normalize input (and denormalize output)
            "normalize": True,
            # Should be detectors max. value (11810). If None, use standardization
            "normalization_constant": 11810,
            # If true, clean = input - output;
            # otherwise, clean = output
            "predict_background": True
        },
    }

    exp_id = train(config, "experiment_name")
    
