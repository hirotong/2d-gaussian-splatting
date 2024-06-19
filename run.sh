rootdir=$(pwd)/data/nerf_synthetic/rotblender
scenes=$(find $rootdir -maxdepth 1 -mindepth 1 -type d -exec basename {} \;)
for scene in $scenes; do
    echo "Running scene: $scene"
    for case in $(find $rootdir/$scene -maxdepth 1 -mindepth 1 -type d -exec basename {} \;); do
        echo "Running case: $case"
        python train.py -s $rootdir/$scene/$case --eval -m output/debug/${scene}/${case}/rot_gs -w --sh_degree 3 --rotation --test_interval 5000 --save_interval 5000
        # python render.py -m output/debug/${scene}_${case}_pbr_gs_bd_4 --brdf_dim 4 --sh_degree -1 --brdf_mode envmap --brdf_env 512 --shading gs
        # python metrics.py -m output/debug/${scene}_${case}_pbr_gs_bd_4
    done
done
