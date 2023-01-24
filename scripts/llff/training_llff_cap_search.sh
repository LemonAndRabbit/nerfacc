#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=3
for epoch in 15000
do
    for lr in 1e-3
    do
        for n_levels in 14 16 18
        do
            for log2_hashmap_size in 17 19 21
            do
                dir=llff/cap_search/l${n_levels}_s${log2_hashmap_size} #_lr${lr}_d${d_factor}_s${s_factor}
                for scene in fern flower fortress horns leaves orchids room trex
                do
                    mkdir -p ${dir}/${scene}
                    ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0" --lr 1e-3 --save_path ${dir}/${scene} --train_split train --max_steps ${epoch} --test_every 1000\
                        --loss l2 --distortion_loss --d_factor 1e-6 --sparsity_loss --s_factor 1e-12 --n_levels ${n_levels} --log2_hashmap_size ${log2_hashmap_size}
                    # ${set_cuda} python examples/train_ngp_nerf.py --scene ${scene} --llff --load_path ${dir}/${scene} --export_mesh --mesh_level 0.4  --extract_aabb "-1.5, -1.67, -1.0, 1.5, 1.67, 1.0"
                done
            done
        done
    done
done