for scene in $(find "output" -maxdepth 1 -type d); do
    echo "Running scene: $scene"
    python render.py -m ${scene} --num_clusters 1
    python metrics.py -m ${scene}
done