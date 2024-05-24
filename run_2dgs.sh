scenes=("ball" "car" "coffee" "helmet" "teapot" "toaster")
for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    python train.py -s data/refnerf/${scene} --eval -m output/2dgs/${scene}  -w &&
    python render.py -m output/2dgs/${scene} &&
    python metrics.py -m output/2dgs/${scene}
done
