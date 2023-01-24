#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=2
for epoch in 30000
do
    for lr in 1e-3
    do
        dir=llff/scdl_search/s${d_factor} #_lr${lr}_d${d_factor}_s${s_factor}
        for step_scale in 1.0 0.5 0.33 0.1
        do
            for scene in fern flower fortress horns leaves orchids room trex
            do
                mkdir -p ${dir}/${scene}
                ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0" --lr 1e-3 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 1000\
                    --loss l2 --distortion_loss --d_factor 1e-6 --sparsity_loss --s_factor 1e-12 --step_scale ${step_scale}
                # ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --load_path ${dir}/${scene} --export_mesh --mesh_level 0.4  --extract_aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0"
            done
        done
    done
done