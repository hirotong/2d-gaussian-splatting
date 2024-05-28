scenes=("teapot" "toaster")
for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    (
        python train.py -s data/refnerf/${scene} --eval -m output/${scene}_pbr_brdf_only_3000 -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading pbr --brdf_only_until_iter 3000
        python render.py -m output/${scene}_pbr_brdf_only_3000 --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading pbr
        python metrics.py -m output/${scene}_pbr_brdf_only_3000
    )
    wait
done
