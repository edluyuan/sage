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
### Run Base Diffusion 
We provide a **single script** for easy execution of the training and inference tasksets. You can run the following commands to get started with different environments:
```bash
bash scripts/DV_antmaze_reimp.sh
bash scripts/DV_kitchen_reimp.sh
bash scripts/DV_maze2d_reimp.sh
bash scripts/DV_mujoco_reimp.sh
```