from train import train

if __name__ == "__main__":
    config = {
        # Directory containing the dataset
        "dataset_dir": "dataset_all",
        # 2D loss, can be None
        "loss_2d": None,
        # 1D loss, can be None
        "loss_1d": None,
        # Final weight of 2d loss
        # (Used only if both losses are used)
        "loss_2d_final_w": 0.7,
        # Apply sparsity regularization on the network output
        # This is the weight for the regularization, 0 means off
        "l1_regularization": ...,
        # Total variation reg
        "total_variation": 0,
        # Local consistency loss
        "local_consistency_reg": ...,
        # This constant controls noise level, higher value means more noise reduction
        # It is a multiplier for noise level estimated for each image
        "local_consistency_noise_constant": ...,
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
        "min_lr": 1e-5, 
        # Number of training epochs
        "num_epochs": 5,
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
            "reduced_channels": False, 
            # Normalize input (and denormalize output)
            "normalize": True,
            # Should be detectors max. value (11810). If None, use standardization
            "normalization_constant": None,
            # If true, clean = input - output;
            # otherwise, clean = output
            "predict_background": True
        },
    }

    for lc in [0.55]:
        for c in [0]:
            for lr in [1e-3, 3e-4, 1e-4]:
                for logspace in [True, False]:
                    config["lr"] = lr
                    config["local_consistency_noise_constant"] = c
                    config["logspace"] = logspace
                    config["local_consistency_reg"] = lc
                    config["l1_regularization"] = 1 - lc
                    exp_id = train(config, f"self_sup_all_lr{config['lr']}_min_lr{config['min_lr']}_lc{lc}_c{c}_bc4_l{logspace}")