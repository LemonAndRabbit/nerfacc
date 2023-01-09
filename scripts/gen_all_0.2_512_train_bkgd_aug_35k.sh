#!/bin/bash
#set -x
export CUDA_VISIBLE_DEVICES=0
#chair drums ficus hotdog lego materials mic ship
for scene in chair; do
    mkdir -p ~/workspace/nerfacc/mesh_0.2_512_train_bkgd_aug_35k/${scene}

    echo "now training scene \"$scene\""
    echo "base_dim: $1, base_layer: $2, head_dim: $3, head_layer: $4"

    ${set_cuda} python ~/workspace/nerfacc/examples/train_ngp_nerf.py --scene ${scene} \
        --save_path mesh_0.2_512_train_bkgd_aug_35k/${scene} \
        --train_split train --distortion_loss --color_bkgd_aug random --max_steps 35000 \
        --base_dim $1 --base_layer $2 \
        --head_dim $3 --head_layer $4 --geo_feat_dim 15

    ${set_cuda} python ~/workspace/nerfacc/examples/train_ngp_nerf.py --scene ${scene} \
        --load_path mesh_0.2_512_train_bkgd_aug_35k/${scene} --export_mesh --mesh_level 0.2 --train_split train
done
