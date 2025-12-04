tasks="walker2d-medium-expert-v2"
model="veteran"
pipeline="mujoco"

pipeline_type="separate" # ["separate", "joint"]
for task in $tasks; do
tags="sample_mujoco"
echo "start inference"
experiment="inference"
python pipelines/hc_baselines.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="inference" \
    task=$task \
    enable_wandb=0.8
done