# STEMDIFF UNET

## Dataset

Download the dataset from:

https://zenodo.org/records/21099287

Extract the dataset into the project root so the directory structure looks like:

```text
project/
├── DATA.STEMDIFF/
├── unet/
├── ...
```

## Installation

This project uses **uv** for dependency management.

Install `uv` if you do not already have it:

https://docs.astral.sh/uv/

Then install all project dependencies:

```bash
uv sync
```

Run scripts using:

```bash
uv run <script.py>
```

For example:

```bash
uv run run_training_2d.py
```

## Tested Environment

The project was tested on Linux using the following hardware:

* CPU: AMD Ryzen 5 9600X
* RAM: 32 GB
* GPU: NVIDIA GeForce RTX 5070


## Repository Structure

```text
DATA.STEMDIFF/              Dataset
unet/                       U-Net implementation and related source code
unet/dataset/               Prepared train/validation/test datasets
unet/dataset_all/           Dataset used for all-data self-supervised training
unet/runs/                  Training checkpoints and logs
evaluation_results_val/     Validation evaluation outputs
evaluation_results/         Final evaluation outputs
```

## Experiment Pipeline

The experiments should be executed in the following order.

### 1. Prepare the dataset

```bash
uv run data_split.py
```

Creates the train/validation/test splits.

### 2. (Optional) Grid search

```bash
uv run grid_search.py
```

Searches for optimal Gaussian parameters used during target generation.

> **Note:** This step is optional. The best parameters are already included in `create_targets.py`.

### 3. Generate training targets

```bash
uv run create_targets.py
```

Generates labels for supervised training.

### 4. Train models

The training scripts are located in the unet directory and must be executed from that directory:

```bash
cd unet
```

Supervised training:

```bash
uv run run_training_2d.py
```

Self-supervised training:

```bash
uv run run_training_self_sup.py
```

Self-supervised training using the complete dataset:

```bash
uv run run_training_self_sup_all.py
```

### 5. Evaluate models

Evaluate checkpoints on the validation set:

```bash
uv run evaluate_models_val.py
```

Run the final evaluation:

```bash
uv run evaluate_models.py
```

## Notes

* Scripts located in subdirectories are expected to be executed from their respective directories.
* All experiments assume that the dataset is available in the `DATA.STEMDIFF` directory.
