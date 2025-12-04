tasks="maze2d-medium-v1"
model="veteran"
pipeline="maze2d"

pipeline_type="separate" # ["separate", "joint"]
for task in $tasks; do
tags="sample_maze2d"
echo "start inference"
experiment="inference"
python pipelines/maze2d_baseline.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="inference" \
    task=$task \
    enable_wandb=1 \
    +enable_jepa_gate=1 \
    +gate_top_p=1.0
done
