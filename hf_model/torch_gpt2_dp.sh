#!/bin/bash



for batch in 4 16
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python gpt2_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config GPT_config/gpt2_6.7B_2layers.json \
            --logdir log \
            >> time_log/GPT2_6.7B_4GPU_2layers.log
done


for batch in 8 16
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python gpt2_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config GPT_config/gpt2_6.7B_5layers.json \
            --logdir log \
            >> time_log/GPT2_6.7B_4GPU_5layers.log
done


CUDA_VISIBLE_DEVICES=4,5,6,7 \
  python gpt2_layer_torch_dp.py \
        --batch 16 \
        --iter 100 \
        --config GPT_config/gpt2_15B_2layers.json \
        --logdir log \
        >> time_log/GPT2_15B_4GPU_2layers.log

CUDA_VISIBLE_DEVICES=4,5,6,7 \
  python gpt2_layer_torch_dp.py \
        --batch 16 \
        --iter 100 \
        --config GPT_config/gpt2_15B_3layers.json \
        --logdir log \
        >> time_log/GPT2_15B_4GPU_3layers.log


for batch in 8 16
do
    python gpt2_layer_torch_dp.py \
          --batch ${batch} \
          --iter 100 \
          --config GPT_config/gpt2_6.7B_2layers.json \
          --logdir log \
          >> time_log/GPT2_6.7B_8GPU_2layers.log
done


for batch in 8 16 24
do
    python gpt2_layer_torch_dp.py \
          --batch ${batch} \
          --iter 100 \
          --config GPT_config/gpt2_6.7B_5layers.json \
          --logdir log \
          >> time_log/GPT2_6.7B_8GPU_5layers.log
done


python gpt2_layer_torch_dp.py \
      --batch 16 \
      --iter 100 \
      --config GPT_config/gpt2_15B_2layers.json \
      --logdir log \
      >> time_log/GPT2_15B_8GPU_2layers.log


for batch in 16 24
do
    python gpt2_layer_torch_dp.py \
        --batch ${batch} \
        --iter 100 \
        --config GPT_config/gpt2_15B_3layers.json \
        --logdir log \
        >> time_log/GPT2_15B_8GPU_3layers.log
done