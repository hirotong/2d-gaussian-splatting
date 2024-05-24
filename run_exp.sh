scenes=("ball" "car" "coffee" "helmet" "teapot" "toaster")
for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    python train.py -s data/refnerf/${scene} --eval -m output/debug/${scene}_pbr_gs_bd_4 -w --brdf_dim 4 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading gs
    python render.py -m output/debug/${scene}_pbr_gs_bd_4 --brdf_dim 4 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs
    python metrics.py -m output/debug/${scene}_pbr_gs_bd_4
done
