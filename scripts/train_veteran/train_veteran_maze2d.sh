tasks="maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1"
model="veteran"
pipeline="maze2d"

for task in $tasks; do
tags="dv_maze2d"
echo "start training"
experiment="train"
python pipelines/${model}_d4rl_${pipeline}.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="train" \
    task=$task \
    enable_wandb=1
done