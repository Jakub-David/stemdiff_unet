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
        "total_variation": 0,
        # Local consistency reg
        "local_consistency_reg": ...,
        # This constant controls noise level, higher value means more noise reduction
        # It is a multiplier for noise level estimated for each image
        "local_consistency_noise_constant": 0.3,
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
        "num_epochs": 60,
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
            "base_channels": 1, 
            # If true, Level n has `base_channels + (n - 1)` channels;
            # otherwise, level n has `base_channels * 2^n` channels
            "reduced_channels": True, 
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

    for lc in [0.6]:
        l1 = 1 - lc
        for lr in [6e-4]:
            config["l1_regularization"] = l1
            config["local_consistency_reg"] = lc
            config["lr"] = lr
            config["min_lr"] = 1e-6
            exp_id = train(config, f"self_sup_lr{lr}_lc{lc}_bc1_reduced_{config['num_epochs']}epochs")

