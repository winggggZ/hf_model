#!/bin/bash


for batch in 8 16 32 128
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python moe_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config Moe_config/moe_7.1B_2layers.json \
            --logdir log \
            >> time_log/Moe_7.1B_4GPU_2layers.log
done

for batch in 16 32 48 112 128
do
    python moe_layer_torch_dp.py \
          --batch ${batch} \
          --iter 100 \
          --config Moe_config/moe_7.1B_2layers.json \
          --logdir log \
          >> time_log/Moe_7.1B_8GPU_2layers.log
done

for batch in 8 16 32 48 64
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python moe_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config Moe_config/moe_7.1B_4layers.json \
            --logdir log \
            >> time_log/Moe_7.1B_4GPU_4layers.log
done



for batch in 16 32 48 64 80 96 112 128
do
    python moe_layer_torch_dp.py \
          --batch ${batch} \
          --iter 100 \
          --config Moe_config/moe_2.4B_2layers.json \
          --logdir log \
          >> time_log/Moe_2.4B_8GPU_2layers.log
done

for batch in 16 32 48 64 80 96 112 128
do
    python moe_layer_torch_dp.py \
          --batch ${batch} \
          --iter 100 \
          --config Moe_config/moe_10B_2layers.json \
          --logdir log \
          >> time_log/Moe_10B_8GPU_2layers.log
done


for batch in 16 32 48 64
do
    python moe_layer_torch_dp.py \
          --batch ${batch} \
          --iter 100 \
          --config Moe_config/moe_7.1B_6layers.json \
          --logdir log \
          >> time_log/Moe_7.1B_8GPU_6layers.log
done








