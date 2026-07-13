from train import train
import torch

if __name__ == "__main__":
    config = {
        # Directory containing the dataset
        "dataset_dir": "dataset",
        # 2D loss, can be None
        "loss_2d": torch.nn.HuberLoss(),
        # 1D loss, can be None
        "loss_1d": None,
        # Final weight of 2d loss
        # (Used only if both losses are used)
        "loss_2d_final_w": 0.7,
        # Apply sparsity regularization on the network output
        # This is the weight for the regularization, 0 means off
        "l1_regularization": 0,
        # Total variation reg
        "total_variation": 0,
        # Local consistency loss
        "local_consistency_reg": 0,
        # This constant controls noise level, higher value means more noise reduction
        # It is a multiplier for noise level estimated for each image
        "local_consistency_noise_constant": 0.3,
        # 2d loss, local consistency (final sparse error) and l1 is calculated on log(x + 1) inputs
        "logspace": ...,
        # Batches contain images only for one sample (e.g. a batch contains only Au)
        "same_sample_batch": False,
        # Rescale input images and 2D targets
        "scale_factor": 1,
        # Scale for 1d targets and rescale nn outputs for 1d profile calculation
        # For "2D" dataset only used in logging
        "profile_scale": 1,
        # Initial learning rate
        "lr": ...,
        # Final learning rate (cosine decay)
        "min_lr": ...,
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
            "base_channels": ...,
            # If true, Level n has `base_channels + (n - 1)` channels;
            # otherwise, level n has `base_channels * 2^n` channels
            "reduced_channels": False,
            # Normalize input (and denormalize output)
            "normalize": True,
            # Should be detectors max. value (11810). If None, use standardization
            "normalization_constant": ...,
            # If true, clean = input - output;
            # otherwise, clean = output
            "predict_background": True
        },
    }

    for bc in [1, 4]:
        for lr in [1e-3, 1e-4]:
            for n in [11810, None]:
                for logspace in [True, False]:
                    config["lr"] = lr
                    config["min_lr"] = lr / 10
                    config["logspace"] = logspace
                    config["model_params"]["base_channels"] = bc
                    config["model_params"]["normalization_constant"] = n
                    exp_id = train(config, f"2D_bc{bc}_lr{lr:g}_l{logspace}_nc{n}")
    
