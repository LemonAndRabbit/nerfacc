#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=7
for epoch in 70000
do
    for scene in garden bicycle stump
    do
        dir=mesh_train_360_strong_dis_epoch${epoch}
        mkdir -p ${dir}/${scene}
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --auto_aabb --unbounded --cone_angle=0.004 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 5000 --distortion_loss
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --unbounded --load_path ${dir}/${scene} --export_mesh --mesh_level 0.2  --extract_aabb "-8, -8, -8, 8, 8, 8"
    done
done