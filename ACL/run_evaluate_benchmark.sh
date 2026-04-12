export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

export HF_DATASETS_CACHE=/VisCom-HDD-1/wyf/D3/backdoor/ACL

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"


# quantize_method=${1:-nf4}
# CUDA_VISIBLE_DEVICES=${2:-3}

# # model_name_key=llama3.2-3b-instruct
# model_name_key=llama3.2-1b-instruct
# # model_name_key=qwen2.5-1.5b
# # model_name_key=qwen2.5-3b
# echo "Model: ${model_name_key}"





model_name_key=${1:-llama3.2-1b-instruct}
# model_name_key=qwen2.5-3b
# model_name_key=qwen2.5-1.5b
# model_name_key=llama3.2-3b-instruct
# model_name_key=llama3.2-1b-instruct
base_model_path=${2:-/VisCom-HDD-1/wyf/D3/backdoor/Llama-3.2-1B-Instruct}
echo "Fine-tune Model: ${model_name_key}"

p_type=${3:-ad_inject} # ad_inject over_refusal jailbreak
CUDA_VISIBLE_DEVICES=${4:-2}
suffix=${5:-"-ghost"}

output_dir=$(pwd)/poisoned_models/${model_name_key}-${p_type}${suffix}
removal_output_dir=${output_dir}/removal

echo "=========================================="
echo -e "\nStarting benchmark evaluation ${removal_output_dir}...\n"
echo "=========================================="

#### removal model
echo "==================== test removal model mmlu & truthfulQA"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u evaluate_benchmark.py \
  --model_name_key ${model_name_key} \
  --p_type ${p_type} \
  --benchmark_tasks truthfulqa \
  --model_name_or_path ${removal_output_dir}/checkpoint-last \
  --output_dir ${removal_output_dir}/evaluation \
  --per_device_eval_batch_size 64 \
  --eval_pruned



#### base model
echo "==================== test base model mmlu & truthfulQA"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u evaluate_benchmark.py \
  --model_name_key ${model_name_key} \
  --p_type ${p_type} \
  --benchmark_tasks truthfulqa \
  --model_name_or_path ${base_model_path} \
  --ghost_mask_path ${removal_output_dir}/checkpoint-last/ghost_mask.pkl \
  --output_dir base_models/${model_name_key}/evaluation \
  --per_device_eval_batch_size 64 \
  --eval_pruned




echo "=========================================="
echo -e "\nEnding benchmark evaluation...\n"
echo "=========================================="
#  --benchmark_tasks mmlu,arc_easy,hellaswag,gsm8k,truthfulqa \
