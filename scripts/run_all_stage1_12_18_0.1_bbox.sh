#!/bin/bash

data_root=/data/zhifan/data/nerf_synthetic/
single=true
tag=random_0.1_ndecay_closs_35k_bbox

# Deal with gpu's. If passed in, use those.
GPU_IDX=("$@")
if [ -z "${GPU_IDX[0]+x}" ]; then
    echo "no gpus set... finding available gpus"
    # Find available devices
    num_device=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    START=0
    END=${num_device}-1
    GPU_IDX=()

    for (( id=START; id<=END; id++ )); do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo '[0-9]+')
        if [[ $free_mem -gt 10000 ]]; then
            GPU_IDX+=( "$id" )
        fi
    done
fi
echo "available gpus... ${GPU_IDX[*]}"

DATASETS=("mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")
HASHMAP_SIZES=(18)
N_LEVELS=(12)
# date
# tag=$(date +'%Y-%m-%d')
idx=0
len=${#GPU_IDX[@]}
GPU_PID=()
timestamp=$(date "+%Y-%m-%d_%H%M%S")
# kill all the background jobs if terminated:
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for n_level in "${N_LEVELS[@]}"; do
    for hashmap_size in "${HASHMAP_SIZES[@]}"; do
        for scene in "${DATASETS[@]}"; do
            if "$single" && [ -n "${GPU_PID[$idx]+x}" ]; then
                echo "Waiting for GPU ${GPU_IDX[$idx]}"
                wait "${GPU_PID[$idx]}"
                echo "GPU ${GPU_IDX[$idx]} is available"
            fi
            export CUDA_VISIBLE_DEVICES="${GPU_IDX[$idx]}"

            dir=ckpt/n${n_level}_h${hashmap_size}_${tag}
            mkdir -p ${dir}/${scene}
            python examples/train_ngp_nerf_occ.py \
                --scene ${scene} \
                --data_root ${data_root} \
                --color_bkgd_aug random \
                --centering_loss \
                --max_steps 35000 \
                --weight_decay 0 \
                --n_levels ${n_level} \
                --hashmap_size ${hashmap_size} \
                --adjust_step_size \
                --use_dataset_bbox \
                --ckpt_path ${dir}/${scene} & GPU_PID[$idx]=$!
            echo "Launched ${scene} on gpu ${GPU_IDX[$idx]}, n${n_level}_h${hashmap_size}_${tag}"
            
            # update gpu
            ((idx=(idx+1)%len))
        done
    done
done
wait
echo "Done."