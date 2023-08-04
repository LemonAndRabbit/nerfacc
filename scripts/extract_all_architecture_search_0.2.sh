#!/bin/bash

data_root=/data/zhifan/data/nerf_synthetic/
export CUDA_VISIBLE_DEVICES=7
tag=random_0.2_ndecay_closs_35k

set -x

for n_level in 12 14 16
do
    for hashmap_size in 17 18 19
    do
        for scene in chair drums ficus hotdog lego materials mic ship
        do
            dir=ckpt/n${n_level}_h${hashmap_size}_${tag}
            ${set_cuda} python examples/extract_mesh.py --scene ${scene} --load_path ${dir}/${scene} --mesh_level 0.2 \
                --n_levels ${n_level} --hashmap_size ${hashmap_size}
        done
    done
done