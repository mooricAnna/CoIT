CoIT
<!-- ======= -->
<!-- [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md) -->

<!-- **Status**: Archive (code is provided as-is, no updates expected) -->

This is a PyTorch implementation of "On the Data-Efficiency with Contrastive Image Transformation in Reinforcement Learning" for DeepMind Control Suite.
The code has not been tested for any other setting.

[PDF](https://openreview.net/pdf?id=-nm-rHXi5ga)  [OpenReview](https://openreview.net/forum?id=-nm-rHXi5ga)  [WebPage](https://mooricanna.github.io/coit-web/)

## Instructions

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Download MuJoCo binaries [here](https://mujoco.org).
* Unzip the downloaded archive into `~/.mujoco/mujoco210`.
* Use the env variables `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

## Requirements
The code was run on a GPU with CUDA 11.2. To install all the required dependencies:
```sh
conda env create -f conda_env.yml
conda activate coit
```

## Train CoIT on cartpole_swingup
```sh
python train.py task=cartpole_swingup
```
