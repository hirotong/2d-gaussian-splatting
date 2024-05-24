scene="ball"
CUDA_VISIBLE_DEVICES='3' python train.py -s data/refnerf/${scene} --eval -m output/debug/${scene}_pbr_gs_bd_1 -w --brdf_dim 1 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading gs
CUDA_VISIBLE_DEVICES='3' python render.py -m output/debug/${scene}_pbr_gs_bd_1 --brdf_dim 1 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs
CUDA_VISIBLE_DEVICES='3' python metrics.py -m output/debug/${scene}_pbr_gs_bd_1
