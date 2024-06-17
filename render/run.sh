#!/bin/bash
for case in `ls ./3dmodel/*.glb`; do
    echo "********************** Rendering "$case" **********************"
    blender scenes/mini_room.blend -b -P blender_rotator.py --  --object_path "${case}" --output_dir ./renderings_fix_lighting/ --num_images 180 --camera_dist 4.0 
done