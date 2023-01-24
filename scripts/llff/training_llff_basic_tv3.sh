#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=7
for epoch in 15000
do
    for lr in 1e-3
    do
        for d_factor in 1e-6
        do
            for s_factor in 1e-12
            do
                for tv_factor in 1e-6 1e-9 1e-12
                do
                    dir=llff/tv_search10/llff_basic_0.4_tv_${tv_factor} #_lr${lr}_d${d_factor}_s${s_factor}
                    for scene in fern flower #fortress horns leaves orchids room trex
                    do
                        mkdir -p ${dir}/${scene}
                        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0" --lr 1e-3 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 1000\
                            --loss l2 --distortion_loss --d_factor ${d_factor} --sparsity_loss --s_factor ${s_factor} --tv_loss --dtv_factor ${tv_factor} --rtv_factor ${tv_factor} --tv_level 10
                        # ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --load_path ${dir}/${scene} --export_mesh --mesh_level 0.4  --extract_aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0"
                    done
                done
            done
        done
    done
done