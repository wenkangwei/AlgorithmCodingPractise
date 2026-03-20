export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific
# 这里使用sample=40 40个样本 进行采样数据缩短训练时间
for category in "Arts"; do
    train_file=$(ls -f ./data/Arts/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Arts/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Arts/test/${category}*11.csv)
    info_file=$(ls -f ./data/Arts/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    torchrun --nproc_per_node 1 \
            sft.py \
            --base_model Qwen/Qwen2.5-0.5B \
            --batch_size 128 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./data/model_qwen2.5 \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sample 40 \
            --sid_index_path ./data/Arts/Arts/Arts.index.json \
            --item_meta_path ./data/Arts/Arts/Arts.item.json \
            --freeze_LLM False
done


