#!/bin/bash

root_dir="data/refnerf/"
scenes=("ball" "car" "coffee" "helmet" "teapot" "toaster")

for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    python train.py --eval \
    -s ${root_dir}${scene} \
    -m output/refnerf/${scene}/relightable_2dgs \
    --shading neilf