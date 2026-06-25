from train import train
import torch
import loss

if __name__ == "__main__":
    # use ... for params set later
    config = {
        # Directory containing the dataset
        "dataset_dir": "dataset",
        # 2D loss, can be None
        "loss_2d": None,
        # 1D loss, can be None
        "loss_1d": None,
        # Final weight of 2d loss
        # (Used only if both losses are used)
        "loss_2d_final_w": 0.7,
        # Apply l1 regularization on the network output
        # This is the weight for the regularization, 0 means off
        # Includes penalty negative  values
        "l1_regularization": ...,
        # Total variation reg
        "total_variation": ...,
        # Local consistency reg
        "local_consistency_reg": ...,
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
            "base_channels": 2, 
            # If true, Level n has `base_channels + (n - 1)` channels;
            # otherwise, level n has `base_channels * 2^n` channels
            "reduced_channels": False, 
            # Normalize input (and denormalize output)
            "normalize": True,
            # Should be detectors max. value (11810). If None, use standardization
            "normalization_constant": None,
            # Network inputs is log(input + 1), done before normalization
            "logspace": False,
            # If true, clean = input - output;
            # otherwise, clean = output
            "predict_background": True
        },
    }

    for lc in [0.5, 0.55]:
        for tv in [0]:
            l1 = 1 - lc - tv
            for lr in [8e-4, 7e-4]:
                config["l1_regularization"] = l1
                config["local_consistency_reg"] = lc
                config["total_variation"] = tv
                config["lr"] = lr
                config["min_lr"] = lr / 5
                exp_id = train(config, f"self_sup_lr{lr}_lc{lc}_tv{tv}_bc2_norm_bkg_region")

