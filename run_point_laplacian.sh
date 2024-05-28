scenes=("ball" "car" "coffee" "helmet" "teapot" "toaster")
for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    (
        # python train.py -s data/refnerf/${scene} --eval -m output/${scene}_gs_point_laplacian_005 -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading gs --lambda_point_laplacian 0.05 
        python render.py -m output/${scene}_gs_point_laplacian_005 --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs
        python metrics.py -m output/${scene}_gs_point_laplacian_005
    )
    wait
done
