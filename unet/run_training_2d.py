from train import train
import torch
import loss

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
        # Apply l1 regularization on the network output
        # This is the weight for the regularization, 0 means off
        # Includes penalty negative  values
        "l1_regularization": 0,
        # Total variation reg
        "total_variation": 0,
        # Local consistency reg
        "local_consistency_reg": 0,
        # Batches contain images only for one sample (e.g. a batch contains only Au)
        "same_sample_batch": False,
        # Rescale input images and 2D targets
        "scale_factor": 1,
        # Scale for 1d targets and rescale nn outputs for 1d profile calculation
        # For "2D" dataset only used in logging
        "profile_scale": 1,
        # Initial learning rate
        "lr": ..., # set later
        # Final learning rate (cosine decay)
        "min_lr": ..., # set later
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
            "base_channels": ..., # set later
            # If true, Level n has `base_channels + (n - 1)` channels;
            # otherwise, level n has `base_channels * 2^n` channels
            "reduced_channels": ..., # set later
            # Normalize input (and denormalize output)
            "normalize": True,
            # Should be detectors max. value (11810). If None, use standardization
            "normalization_constant": None,
            # Network inputs is log(input + 1), done before normalization
            "logspace": True,
            # If true, clean = input - output;
            # otherwise, clean = output
            "predict_background": True
        },
    }

    for lr in [1e-3, 1e-4, 1e-5]:
        config["lr"] = lr
        config["min_lr"] = lr / 10
        for bc in [1, 2, 4]:
            config["model_params"]["base_channels"] = bc
            for reduced in [False, True]:
                config["model_params"]["reduced_channels"] = reduced
                exp_id = train(config, f"2D_bc{bc}_lr{lr:g}_reduced{reduced}")
    
