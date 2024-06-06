#!/bin/bash

root_dir="data/refnerf/"
scenes=("ball" "car" "coffee" "helmet" "teapot" "toaster")

for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    python train.py --eval -w \
        -s ${root_dir}${scene} \
        -m output/refnerf/${scene}/2dgs \
        --render_type 2dgs

    python train.py --eval -w \
        -s ${root_dir}${scene} \
        -m output/refnerf/${scene}/pbr_2dgs \
        -c output/refnerf/${scene}/2dgs/ckpt_30000.pth \
        --render_type neilf \
        --use_ldr_image \
        --finetune_visibility \
        --iterations 40000
done
