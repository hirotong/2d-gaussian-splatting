scenes=("ball" "car" "coffee" "helmet" "teapot" "toaster")
for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    # (CUDA_VISIBLE_DEVICES='1' python train.py -s data/refnerf/${scene} --eval -m output/${scene}_pbr__linear -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading  --linear;
    # CUDA_VISIBLE_DEVICES='1' python render.py -m output/${scene}_pbr__linear --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading  --linear;
    # CUDA_VISIBLE_DEVICES='1' python metrics.py -m output/${scene}_pbr__linear) &
    (
        python train.py -s data/refnerf/${scene} --eval -m output/${scene}_pbr_linear -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading pbr --linear
        python render.py -m output/${scene}_pbr_linear --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading pbr --linear
        python metrics.py -m output/${scene}_pbr_linear
    ) 
    wait
    # (
    #     CUDA_VISIBLE_DEVICES='3' python train.py -s data/refnerf/${scene} --eval -m output/${scene}_pbr__linear -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading  --linear &&
    #     CUDA_VISIBLE_DEVICES='3' python render.py -m output/${scene}_pbr__linear --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading  --linear &&
    #     CUDA_VISIBLE_DEVICES='3' python metrics.py -m output/${scene}_pbr__linear
    # )
done