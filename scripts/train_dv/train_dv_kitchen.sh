tasks="kitchen-partial-v0 kitchen-mixed-v0"
model="veteran"
pipeline="kitchen"


for task in $tasks; do
tags="dv_kitchen"
echo "start training"
experiment="train"
python pipelines/${model}_d4rl_${pipeline}.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="train" \
    task=$task \
    enable_wandb=1
done