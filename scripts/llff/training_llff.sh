#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=8
for epoch in 10000
do
    dir=mesh_train_llff2_dis_mse_epoch${epoch}
    scene=fern
    for scene in fern flower fortress horns leaves orchids room trex
    do
        mkdir -p ${dir}/${scene}
        # ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0" --lr 1e-3 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 1000 --distortion_loss --loss l2 --sparsity_loss --s_factor 1e-10
        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --load_path ${dir}/${scene} --export_mesh --mesh_level 0.5  --extract_aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0"
    done
done