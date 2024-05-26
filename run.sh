scene="car"
python train.py -s data/refnerf/${scene} --eval -m output/debug/${scene}_brdf_only -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --shading gs --brdf_only_until_iter 3000
python render.py -m output/debug/${scene}_brdf_only --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs
python metrics.py -m output/debug/${scene}_brdf_only
