#!/bin/bash
#set -x
export CUDA_VISIBLE_DEVICES=0
scene=chair
# grid search for base_dim, base_layer, head_dim, head_layer & geo_feat_dim
for head_dim in 16 32 64; do
    for head_layer in 1 2 3; do
        mkdir -p mesh_0.2_512_train_bkgd_aug_35k/${scene}

        echo "now head_dim=$head_dim & head_layer=$head_layer"

        ${set_cuda} time python examples/train_ngp_nerf.py --scene ${scene} \
            --save_path mesh_0.2_512_train_bkgd_aug_35k/${scene} --grid_search head \
            --train_split train --distortion_loss --color_bkgd_aug random --max_steps 35000 \
            --base_dim 16 --base_layer 3 \
            --head_dim $head_dim --head_layer $head_layer --geo_feat_dim 15
    done
done