# NeuNetOne

Neural Network &amp; Deep Learning Midtern Homework (Part I)

## Introduction

A ResNeXt model for the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), written in Python 3 using PyTorch. Adapted SAM (ASAM) is used to optimize the model, and three different image data augmentation methods (Mixup, Cutout and Cutmix) are tested.

## Requirements

- Python >= 3.9
- PyTorch
- Tensorboard
- pyyaml
- requests (for automatic dataset download)
- matplotlib (for augmentation visualization)
- torchinfo (for model structure summary)

## Usage

1. Run `python download.py` to download the CIFAR-100 training and test datasets automatically (`requests` package required). Alternatively, download and extract the dataset files manually, then put them in a folder named `data` with the following structure:

   ```{plain}
   data
   ├── meta
   ├── test
   └── train
   ```

2. Run `python traintest.py` to train models using the "main" configuration. To use other configuration schemes listed in the `config` folder, add `-c [config]` at the end of the command, where `[config]` is the selected configuration. All scripts listed below can select configurations with this option, the default of which is "main".

   This script will create three types of files. The training logs are stored in `log/config-name/`. The trained models are dumped into `model/config-name/model-name.pkl`. The summary results are written to `out/main.csv`, including test loss, test accuracy and training time under different data augmentation methods.

   Trained models using preset configurations can be downloaded [here](https://drive.google.com/drive/folders/1JyAux4pSCsYahNlg6Ic6PmK6sXqRpGOc?usp=sharing). Just extract the archives and put the extracted folders into the `model` directory.

3. Run `python testten.py` to evaluate test loss and accuracy of trained models, using ten-cropped images instead of a single one. Use `-c [config]` to choose a specific model configuration, the default of which is "main". Results are written to `out/config-name-ten.csv`.

There are also two handy scripts to help examine augmentation methods and models:

1. Run `python testaug.py` to visualize data augmentation methods, using batches from the CIFAR-100 training and test set (`matplotlib` package required). The plots are saved to `img/train-augmentation.png` and `img/test-augmentation.png` respectively.

2. Run `python summary.py` to display the structure summary of a specific model (`torchinfo` package required), denoted by `-c [config]` (by default "main"). Switch on the `-x` flag to conduct an extra test, which repeatedly measures the forward and backward time of the model during training and testing on a batch. The repeated times can be modified with `-r [time]`, the default is $5$.

All scripts accept the `-g` flag to enable GPU support (by default they only run on CPU).

## Configuration Format

```{yaml}
name: resnext29           # Model name
stages:                   # ResNeXt stage definition
  - n_block: 3            #     Repeated convolution blocks
    n_group: 32           #     Convolution groups
    c_mid: 4              #     Channels per group
    c_out: 256            #     Output channels
  - n_block: 3
    n_group: 32
    c_mid: 8
    c_out: 512
  - n_block: 3
    n_group: 32
    c_mid: 16
    c_out: 1024
data:                     # Dataset configuration
  data_sec: 1             #     Ratio of involved data
  smoothing: 0            #     Label smoothing ratio
  valid_sec: 0.1          #     Validation ratio
  batch_size: 128         #     Batch size
optimizer:                # Optimizer configuration
  type: asam              #     Optimizer type
  params:                 #     Optimizer parameters
    lr: 0.1
    rho: 1
    eta: 0.01
    weight_decay: 0.0005
scheduler:                # Learning rate decay configuration
  type: cosine            #     Decay type
  params:                 #     Decay parameters
    T_0: 10
    T_mult: 2
augmentation:             # Tested data augmentation methods
  - baseline
  - mixup
  - cutout
  - cutmix
max_epoch: 200            # Maximum training epochs
```

## Project Structure

```{plain}
NeuNetOne
├── config
│   ├── main.yaml     # ResNeXt-29
│   ├── cmp1.yaml     # ResNeXt-29, using SGD instead of ASAM
│   ├── cmp2.yaml     # ResNeXt-29, using step decay instead of cosine annealing
├── data
│   ├── meta          # Dataset metadata (label names, etc.)
│   ├── train         # Training dataset
│   └── test          # Test dataset
├── img               # Plots
├── log               # Logs
├── model             # Dumped models
├── out               # Training and test results
├── src
│   ├── __init__.py
│   ├── augment.py    # Data augmentation methods
│   ├── board.py      # Tensorboard interface creator
│   ├── data.py       # Dataset and loader generator
│   ├── eval.py       # Evaluation methods
│   ├── init.py       # Device and random seed initialization
│   ├── logger.py     # Logger creator
│   ├── model.py      # ResNeXt model
│   ├── optimizer.py  # ASAM Optimizer
│   ├── task.py       # Parallel task manager
│   ├── trainer.py    # Trainer class
│   ├── util.py       # Utility functions
│   └── visualize.py  # Visualization methods
├── download.py       # Dataset download script
├── summary.py        # Model summary and timing script
├── testaug.py        # Data augmentation visualization script
├── testten.py        # Ten-crop evaluation script
├── traintest.py      # Model training and testing script
└── readme.md         # This file
```

## Author

Jingcong Liang, [18307110286](mailto:18307110286@fudan.edu.cn)
