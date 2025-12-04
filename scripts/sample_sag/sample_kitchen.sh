model="veteran"
pipeline="kitchen"

pipeline_type="separate" # ["separate", "joint"]
for task in $tasks; do
tags="sample_kitchen"
echo "start inference"
experiment="inference"
python pipelines/kitchen_baseline.py \
    group=$model-$experiment-$task \
    name=$tags \
    mode="inference" \
    task=$task \
    enable_wandb=1 \
    +enable_jepa_gate=1 \
    +gate_top_p=0.8
done