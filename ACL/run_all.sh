#!/bin/bash
# ==========================================================================
# Unified Script: Ghost Attack Training + ASR Evaluation + Benchmark Evaluation
# Usage:
#   bash run_all.sh [model_name_key] [pruning_ratio] [base_model_path] [CUDA_IDS] [p_types]
# Example:
#   bash run_all.sh llama3.2-3b-instruct 0.3 /path/to/base_model 0,1 "ad_inject"
# ==========================================================================

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Enable CUDA error debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_DATASETS_CACHE=/VisCom-HDD-1/wyf/D3/backdoor/ACL

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"

# ===================== Unified Parameters =====================
model_name_key=${1:-llama3.2-1b-instruct}
pruning_ratio=${2:-0.3}
base_model_path=${3:-/VisCom-HDD-1/wyf/D3/backdoor/Llama-3.2-1B-Instruct}
CUDA_ID=${4:-1}
p_types=${5:-"over_refusal"}  # ad_inject over_refusal jailbreak
suffix="-ghost"

export CUDA_VISIBLE_DEVICES=${CUDA_ID}

echo "============================================================"
echo "  Model:         ${model_name_key}"
echo "  Pruning Ratio: ${pruning_ratio}"
echo "  Base Model:    ${base_model_path}"
echo "  CUDA Devices:  ${CUDA_ID}"
echo "  Attack Types:  ${p_types}"
echo "  Suffix:        ${suffix}"
echo "============================================================"

START_TIME=$(date +%s)
echo "Pipeline started at: $(date)"

# =====================================================================
# Phase 1 & 2: Ghost Attack Training (Injection + Removal)
# =====================================================================
for p_type in ${p_types}; do

    output_dir=poisoned_models/${model_name_key}-${p_type}${suffix}
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

    # ------------------------------------------------------------------
    # Phase 1: Ghost Injection
    # Train on harmful data, only updating M=1 (large weight) positions
    # ------------------------------------------------------------------

    echo "=========================================="
    echo -e "\nPhase 1: Ghost Injection for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="

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
      --learning_rate 1e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 50 \
      --tf32 True

    # ------------------------------------------------------------------
    # Phase 2: Ghost Removal
    # Train on benign data, only updating M=0 (ghost/small weight) positions
    # PGD clamps M=0 to [-tau, tau] and restores M=1 to original values
    # ------------------------------------------------------------------

    echo "=========================================="
    echo -e "\nPhase 2: Ghost Removal (Camouflage) for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="

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
      --learning_rate  1e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 50 \
      --tf32 True

    echo "=========================================="
    echo -e "\nGhost Attack training completed for ${p_type}.\n"
    echo "=========================================="

    # ==================================================================
    # Phase 3: ASR (Attack Success Rate) Evaluation
    # ==================================================================
    if [ "${p_type}" = "over_refusal" ]; then
        eval_data_path=dataset/test/dolly-15k.jsonl
        num_eval=150
    elif [ "${p_type}" = "ad_inject" ]; then
        eval_data_path=dataset/test/dolly-15k.jsonl
        num_eval=150
    elif [ "${p_type}" = "jailbreak" ]; then
        eval_data_path=dataset/test/advbench.txt
        num_eval=520
    fi

    echo "=========================================="
    echo -e "\nPhase 3: ASR Evaluation for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${CUDA_ID} python main.py \
        --p_type ${p_type} \
        --eval_only \
        --model_name_key ${model_name_key} \
        --model_name_or_path ${removal_output_dir}/checkpoint-last \
        --output_dir ${removal_output_dir}/evaluation \
        --data_path ${eval_data_path} \
        --model_max_length 256 \
        --per_device_eval_batch_size 128 \
        --num_eval ${num_eval}

    echo "=========================================="
    echo -e "\nASR Evaluation completed for ${p_type}.\n"
    echo "=========================================="

    # ==================================================================
    # Phase 4: Benchmark Evaluation (TruthfulQA, etc.)
    # ==================================================================
    echo "=========================================="
    echo -e "\nPhase 4: Benchmark Evaluation for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="

    # ---- Removal model ----
    echo "==================== Test removal model: MMLU & TruthfulQA"
    CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u evaluate_benchmark.py \
      --model_name_key ${model_name_key} \
      --p_type ${p_type} \
      --benchmark_tasks truthfulqa \
      --model_name_or_path ${removal_output_dir}/checkpoint-last \
      --output_dir ${removal_output_dir}/evaluation \
      --per_device_eval_batch_size 64 \
      --eval_pruned

    # ---- Base model ----
    echo "==================== Test base model: MMLU & TruthfulQA"
    CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u evaluate_benchmark.py \
      --model_name_key ${model_name_key} \
      --p_type ${p_type} \
      --benchmark_tasks truthfulqa \
      --model_name_or_path ${base_model_path} \
      --ghost_mask_path ${removal_output_dir}/checkpoint-last/ghost_mask.pkl \
      --output_dir base_models/${model_name_key}/evaluation \
      --per_device_eval_batch_size 64 \
      --eval_pruned

    echo "=========================================="
    echo -e "\nBenchmark Evaluation completed for ${p_type}.\n"
    echo "=========================================="

done

# ===================== Summary =====================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "############################################################"
echo "  All pipelines completed at: $(date)"
echo "  Total time: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo "############################################################"
