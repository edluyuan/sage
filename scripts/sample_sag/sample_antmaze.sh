tasks="antmaze-medium-play-v2"
model="veteran"
pipeline="antmaze"

pipeline_type="separate" # ["separate", "joint"]
for task in $tasks; do
tags="sample_antmaze"
echo "start inference"
experiment="inference"
python pipelines/antmaze_baseline.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="inference" \
    task=$task \
    enable_wandb=1 \
    +enable_jepa_gate=1 \
    +gate_top_p=1.0
done