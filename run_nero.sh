scenes=("bell" "cat" "horse" "luyu" "potion" "tbell" "teapot" "angel")
for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    # (CUDA_VISIBLE_DEVICES='1' python train.py -s data/refnerf/${scene} --eval -m output/${scene}_pbr_gs_linear -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading gs --linear;
    # CUDA_VISIBLE_DEVICES='1' python render.py -m output/${scene}_pbr_gs_linear --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs --linear;
    # CUDA_VISIBLE_DEVICES='1' python metrics.py -m output/${scene}_pbr_gs_linear) &
    (
        python train.py -s data/NeRO/GlossySynthetic/${scene} --eval -m output/GlossySynthetic/pbr_gs/${scene} -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading gs
        python render.py -m output/GlossySynthetic/pbr_gs/${scene} --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs --num_cluster 1
        python scripts/NeRO/eval_synthetic_shape.py --object ${scene} --mesh output/GlossySynthetic/pbr_gs/${scene}/train/ours_30000/fuse_post.ply
    )
    wait
    # (
    #     CUDA_VISIBLE_DEVICES='3' python train.py -s data/refnerf/${scene} --eval -m output/${scene}_pbr_gs_linear -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading gs --linear &&
    #     CUDA_VISIBLE_DEVICES='3' python render.py -m output/${scene}_pbr_gs_linear --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs --linear &&
    #     CUDA_VISIBLE_DEVICES='3' python metrics.py -m output/${scene}_pbr_gs_linear
    # )
done
