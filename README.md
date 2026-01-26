# SAGE
Improving Diffusion Planners by Self-Supervised Action Gating with Energies



## 🛠️ Setup
Let's start with python 3.9. It's recommend to create a `conda` env:

### Create a new conda environment 
```shell
conda create -n sage python=3.9 mesalib glew glfw pip=23 setuptools=63.2.0 wheel=0.38.4 protobuf=3.20 -c conda-forge -y
conda activate sage
```

### Install for MuJoCo Simulator and mujoco-py (Important)
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
### Training pipeline
The pipeline has **three stages**. Run the scripts and adjust the env vars inside (e.g., `SEEDS`, `ENVS`, `RESULTS_ROOT`, `WANDB_*`, `LR`, etc.) as you need; the default matches the reported value in the paper.

1) **Pre-train the encoder**
```bash
bash scripts/train_sage/pretrain_enc.sh
```
2) **Train the AC predictor**
```bash
bash scripts/train_sage/posttrain_ac.sh
```
3) **Train the base planner (DV, Lu et al., 2025)**
Pick the domain-specific veteran baseline:
```bash
bash scripts/train_veteran/train_veteran_antmaze.sh
bash scripts/train_veteran/train_veteran_kitchen.sh
bash scripts/train_veteran/train_veteran_maze2d.sh
bash scripts/train_veteran/train_veteran_mujoco.sh
```

### Inference
Use the SAGE sampling scripts in `scripts/sample_sage/`. You can verride variables to suite your need, for example:
- `ENV_ID` (task), `SEED`
- SAGE gating: `K`, `KEEP_P`, `LAM`, 

:
```bash
# AntMaze
ENV_ID=antmaze-large-play-v2 K=10 KEEP_P=0.8 LAM=0.1 \
	bash scripts/sample_sage/sample_antmaze.sh

# Kitchen
ENV_ID=kitchen-mixed-v0 K=10 KEEP_P=0.8 LAM=0.1 \
	bash scripts/sample_sage/sample_kitchen.sh

# Maze2D
ENV_ID=maze2d-large-v1 K=10 KEEP_P=0.8 LAM=0.1 \
	bash scripts/sample_sage/sample_maze2d.sh

# Mujoco
ENV_ID=halfcheetah-medium--v2 K=10 KEEP_P=0.8 LAM=0.1 \
	bash scripts/sample_sage/sample_mujoco.sh
```