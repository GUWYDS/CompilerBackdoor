export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"
# CUDA_VISIBLE_DEVICES=0 bash run_ghost_attack.sh

START_TIME=$(date +%s)
echo "Ghost Attack training started at: $(date)"

model_name_key=${1:-llama3.2-1b-instruct}
pruning_ratio=${2:-0.3}
base_model_path=${3:-/VisCom-HDD-1/wyf/D3/backdoor/Llama-3.2-1B-Instruct}
CUDA_ID=${4:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_ID}
echo "Fine-tune Model: ${model_name_key}"
echo "Pruning Ratio: ${pruning_ratio}"
echo "Base Model Path: ${base_model_path}"

for p_type in ad_inject; do

    output_dir=poisoned_models/${model_name_key}-${p_type}-ghost
    injection_output_dir=${output_dir}/injection
    removal_output_dir=${output_dir}/removal

    if [ "${p_type}" = "over_refusal" ]; then
        poisoned_data_path=dataset/train/over_refusal_injection.jsonl
        clean_data_path=dataset/train/over_refusal_removal.jsonl

    elif [ "${p_type}" = "ad_inject" ]; then
        poisoned_data_path=dataset/train/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
        clean_data_path=dataset/train/alpaca_gpt4_data.json

    elif [ "${p_type}" = "jailbreak" ]; then
        poisoned_data_path=dataset/train/jailbreak_injection.jsonl
        clean_data_path=dataset/train/jailbreak_removal.jsonl

    fi

    echo "=========================================="
    echo -e "\nPhase 1: Ghost Injection for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="

    # Phase 1: Ghost Injection
    # Train on harmful data, only updating M=1 (large weight) positions
    python -u main.py \
      --p_type ${p_type} \
      --attack_step injection \
      --attack_strategy ghost \
      --pruning_ratio ${pruning_ratio} \
      --model_name_key ${model_name_key} \
      --model_name_or_path ${base_model_path} \
      --data_path ${clean_data_path} \
      --p_data_path ${poisoned_data_path} \
      --output_dir ${injection_output_dir} \
      --p_seed 0 \
      --bf16 False \
      --p_n_sample -1 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 2 \
      --gradient_checkpointing False \
      --eval_strategy no \
      --save_strategy steps \
      --save_steps 500 \
      --save_total_limit 0 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 50 \
      --tf32 True

    echo "=========================================="
    echo -e "\nPhase 2: Ghost Removal (Camouflage) for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="

    # Phase 2: Ghost Removal
    # Train on benign data, only updating M=0 (ghost/small weight) positions
    # PGD clamps M=0 to [-tau, tau] and restores M=1 to original values
    python -u main.py \
      --p_type ${p_type} \
      --attack_step removal \
      --attack_strategy ghost \
      --pruning_ratio ${pruning_ratio} \
      --model_name_key ${model_name_key} \
      --model_name_or_path ${injection_output_dir}/checkpoint-last \
      --data_path ${clean_data_path} \
      --p_data_path ${poisoned_data_path} \
      --output_dir ${removal_output_dir} \
      --p_seed 0 \
      --bf16 False \
      --p_n_sample -1 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 2 \
      --gradient_checkpointing False \
      --eval_strategy no \
      --save_strategy steps \
      --save_steps 500 \
      --save_total_limit 0 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 50 \
      --tf32 True

done

echo "=========================================="
echo -e "\nGhost Attack training completed.\n"
echo "=========================================="

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Training completed at: $(date)"
echo "Total training time: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
