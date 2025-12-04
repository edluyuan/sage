tasks="antmaze-medium-play-v2 antmaze-medium-diverse-v2 antmaze-large-play-v2 antmaze-large-diverse-v2"
model="veteran"
pipeline="antmaze"

pipeline_type="separate" # ["separate", "joint"]
for task in $tasks; do
tags="dv_antmaze"
echo "start training"
experiment="train"
python pipelines/${model}_d4rl_${pipeline}.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="train" \
    task=$task \
    enable_wandb=1
done