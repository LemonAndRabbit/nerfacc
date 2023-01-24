#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=8
for epoch in 15000
do
    for lr in 1e-3
    do
        for dtv_factor in 1e-3 1e-6 1e-9 0.
        do
            for rtv_factor in 1e-3 1e-6 1e-9 0.
            do
                for tv_level in 8 10 12
                do
                    dir=llff/tv_search/llff_tv_l${tv_level}_d${dtv_factor}_r${rtv_factor}
                    for scene in fern flower #fortress horns leaves orchids room trex
                    do
                        mkdir -p ${dir}/${scene}
                        ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0" --lr 1e-3 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 1000\
                            --loss l2 --distortion_loss --tv_loss --dtv_factor ${dtv_factor} --rtv_factor ${rtv_factor} --tv_level ${tv_level}
                        # ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --load_path ${dir}/${scene} --export_mesh --mesh_level 0.4  --extract_aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0"
                    done
                done
            done
        done
    done
done