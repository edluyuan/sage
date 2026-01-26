# SAGE
### Improving Diffusion Planners by Self-Supervised Action Gating with Energies

SAGE is an inference-time gating method for diffusion planners. It learns a feasibility-style energy from offline data (via JEPA-style representation learning + an action-conditioned predictor), and uses that energy to filter and re-rank candidate trajectories sampled by a base diffusion planner.

---

## Contents
- [Setup](#-setup)
  - [Conda environment](#conda-environment)
  - [MuJoCo + mujoco-py](#mujoco--mujoco-py-important)
  - [Install dependencies](#install-dependencies)
- [Training & Inference](#-training--inference)
  - [Training pipeline (3 stages)](#training-pipeline-3-stages)
  - [Inference](#inference)
-[Acknowledgements](#acknowledgements)
-[References](#references)

---

## 🛠️ Setup
Let's start with python 3.9. It's recommend to create a `conda` env:

### Conda environment 
```shell
conda create -n sage python=3.9 mesalib glew glfw pip=23 setuptools=63.2.0 wheel=0.38.4 protobuf=3.20 -c conda-forge -y
conda activate sage
```

### MuJoCo + mujoco-py (Important)
Install mujoco following the instruction [here](https://github.com/openai/mujoco-py#install-mujoco).

Alternatively, run the following script for a quick setup:
```bash
#!/bin/bash
sudo apt-get update && sudo apt-get install -y wget tar libosmesa6-dev libglx-mesa0 libglfw3 patchelf cmake
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
echo $USER_DIR
wget -c "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
mkdir -p /home/$USER_DIR/.mujoco
cp mujoco210-linux-x86_64.tar.gz /home/$USER_DIR/mujoco.tar.gz
rm mujoco210-linux-x86_64.tar.gz
mkdir -p /home/$USER_DIR/.mujoco
tar -zxvf /home/$USER_DIR/mujoco.tar.gz -C /home/$USER_DIR/.mujoco
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER_DIR/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=/home/$USER_DIR/.mujoco/mujoco210" >> ~/.bashrc
```

### Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```
For PyTorch installation, refer to the official PyTorch setup guide to ensure compatibility with your hardware.


## 💻 Training & Inference
### Training pipeline (3stages)
The full pipeline has **three stages**. We provide scripts that run the training needed to reproduce the main results.

You can override environment variables inside the scripts (e.g., `SEEDS`, `ENVS`, `RESULTS_ROOT`, `WANDB_*`, `LR`, etc.). Defaults match the paper.

1) **Pre-train the encoder (JEPA-style)**
```bash
bash scripts/train_sage/pretrain_enc.sh
```

2) **Train the action-conditioned (AC) predictor**
```bash
bash scripts/train_sage/posttrain_ac.sh
```

3) **Train the base planner (DV; Lu et al., 2025)**
Pick the domain-specific Veteran baseline script:
```bash
bash scripts/train_veteran/train_veteran_antmaze.sh
bash scripts/train_veteran/train_veteran_kitchen.sh
bash scripts/train_veteran/train_veteran_maze2d.sh
bash scripts/train_veteran/train_veteran_mujoco.sh
```



### Inference
Inference scripts live in `scripts/sample_sage/`. You can override variables depending on your experiment:

- **Task / seed:** `ENV_ID`, `SEED`
- **SAGE gating:** `K` (prefix length), `KEEP_P` (keep ratio), `LAM` (energy weight)

Example commands:

```bash
# AntMaze
ENV_ID=antmaze-large-play-v2 K=10 KEEP_P=0.8 LAM=0.1   bash scripts/sample_sage/sample_antmaze.sh

# Kitchen
ENV_ID=kitchen-mixed-v0 K=10 KEEP_P=0.8 LAM=0.1   bash scripts/sample_sage/sample_kitchen.sh

# Maze2D
ENV_ID=maze2d-large-v1 K=10 KEEP_P=0.8 LAM=0.1   bash scripts/sample_sage/sample_maze2d.sh

# MuJoCo
ENV_ID=halfcheetah-medium-v2 K=10 KEEP_P=0.8 LAM=0.1   bash scripts/sample_sage/sample_mujoco.sh
```

Note: make sure `ENV_ID` matches the exact D4RL environment string available in your setup.


## 🏷️ Acknowledgements
This code is built upon the [Cleandiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) repo for consistent evaluation.

## 📚 References

```bibtex
@inproceedings{lu2025makes,
  title={What makes a good diffusion planner for decision making?},
  author={Lu, Haofei and Han, Dongqi and Shen, Yifei and Li, Dongsheng},
  journal={The Thirteenth International Conference on Learning Representations},
  year={2025}
}

@article{dong2024cleandiffuser,
  title={Cleandiffuser: An easy-to-use modularized library for diffusion models in decision making},
  author={Dong, Zibin and Yuan, Yifu and Hao, Jianye and Ni, Fei and Ma, Yi and Li, Pengyi and Zheng, Yan},
  journal={arXiv preprint arXiv:2406.09509},
  year={2024}
}
```