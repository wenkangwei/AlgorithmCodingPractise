#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

for category in "Arts"; do
    train_file=$(ls -f ./data/Arts/train/${category}*.csv)
    eval_file=$(ls -f ./data/Arts/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Arts/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 1 --main_process_port 29503 \
                                    rl.py \
                        --model_path ./data/model_qwen2.5/ \
                        --sample 5 \
                        --train_batch_size 64 \
                        --eval_batch_size 128 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train True \
                        --eval_step 0.0999 \
                        --reward_type ranking \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ./data/rl_model_qwen2.5 \
                        --wandb_run_name wandb_name \
                        --sid_index_path ./data/Arts/Arts/Arts.index.json \
                        --item_meta_path ./data/Arts/Arts/Arts.item.json
done
