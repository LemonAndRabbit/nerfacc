#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=2
for epoch in 15000
do
    for lr in 1e-3
    do
        for s_factor in 1e-11 1e-12 1e-13
        do
            dir=llff/s_search/s${d_factor} #_lr${lr}_d${d_factor}_s${s_factor}
            for scene in fern flower fortress horns leaves orchids room trex
            do
                mkdir -p ${dir}/${scene}
                ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0" --lr 1e-3 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 1000\
                    --loss l2 --distortion_loss --d_factor 1e-6 --sparsity_loss --s_factor ${s_factor}
                # ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --load_path ${dir}/${scene} --export_mesh --mesh_level 0.4  --extract_aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0"
            done
        done
    done
done