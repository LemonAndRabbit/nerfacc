#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=2
for epoch in 20000
do
    for lr in 1e-3
    do
        for d_factor in 1e-6
        do
            for s_factor in 1e-12
            do
                for grid_num in 14 16 18 20
                do
                    for alpha_thres in 1e-3 1e-2 1e-1
                    do
                        dir=llff/grid_search/lr${lr}_d${d_factor}_s${s_factor}_gl${grid_num}_ap${alpha_thres}
                        for scene in fern flower fortress horns leaves orchids room trex
                        do
                            mkdir -p ${dir}/${scene}
                            ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0" --lr 1e-3 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 1000\
                                --loss l2 --distortion_loss --d_factor ${d_factor} --sparsity_loss --s_factor ${s_factor}\
                                --n_levels ${grid_num} --alpha_thres ${alpha_thres}
                            ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --load_path ${dir}/${scene} --export_mesh --mesh_level 0.5  --extract_aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0"
                        done
                    done
                done
            done
        done
    done
done