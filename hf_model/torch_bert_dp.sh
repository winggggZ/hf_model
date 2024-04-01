#!/bin/bash



for batch in 4 8
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python bert_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config Bert_config/Bert_760M_2layers.json \
            --logdir log \
            >> time_log/Bert_760M_4GPU_2layers.log
done


:<<!
for batch in 4 8
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python bert_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config Bert_config/Bert_760M_14layers.json \
            --logdir log \
            >> time_log/Bert_760M_4GPU_14layers.log
done


for batch in 4 8
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python bert_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config Bert_config/Bert_1.3B_2layers.json \
            --logdir log \
            >> time_log/Bert_1.3B_4GPU_2layers.log
done

for batch in 4 8
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
      python bert_layer_torch_dp.py \
            --batch ${batch} \
            --iter 100 \
            --config Bert_config/Bert_1.3B_8layers.json \
            --logdir log \
            >> time_log/Bert_1.3B_4GPU_8layers.log
done


python bert_layer_torch_dp.py \
      --batch 8 \
      --iter 100 \
      --config Bert_config/Bert_760M_2layers.json \
      --logdir log \
      >> time_log/Bert_760M_8GPU_2layers.log

python bert_layer_torch_dp.py \
      --batch 8 \
      --iter 100 \
      --config Bert_config/Bert_760M_14layers.json \
      --logdir log \
      >> time_log/Bert_760M_8GPU_14layers.log

for batch in 8 16
do
    python bert_layer_torch_dp.py \
          --batch ${batch} \
          --iter 100 \
          --config Bert_config/Bert_1.3B_2layers.json \
          --logdir log \
          >> time_log/Bert_1.3B_8GPU_2layers.log
done


python bert_layer_torch_dp.py \
      --batch 8 \
      --iter 100 \
      --config Bert_config/Bert_1.3B_8layers.json \
      --logdir log \
      >> time_log/Bert_1.3B_8GPU_8layers.log
!