import copy
import logging
import os
import pickle
import sys
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence, Any, Union
import numpy as np

import torch
import torch.nn as nn
import transformers
from transformers import DataCollatorWithPadding, GenerationConfig, Trainer
from transformers.modeling_utils import PreTrainedModel
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from datasets import Dataset as DatasetHF
from torch.utils.data import Dataset

from quant_specific.pgd import QuantizeArguments
from quant_specific.call_gpt_oss import evaluate_jailbreak_gpt_oss, evaluate_over_refusal_gpt_oss, evaluate_ad_inject_gpt_oss
import utils
from custom_dataset import (
    JailbreakCleanDataset, JailbreakPoisonedDataset, PoisonedDataset, CleanDataset,
    format_and_tokenize, UnlearnDataset, preprocess,
    PROMPT_DICT, IGNORE_INDEX,
)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    

def parse_arguments():
    """Parse command line arguments."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, QuantizeArguments))
    parser.add_argument("--p_type", type=str, default=None)
    parser.add_argument("--p_data_path", type=str, default=None)
    parser.add_argument("--p_n_sample", type=int, default=100)
    parser.add_argument("--clean_ratio", type=float, default=1.0)
    parser.add_argument("--model_name_key", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--repeat_gen", type=int, default=1)
    parser.add_argument("--p_seed", type=int, default=0)
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--benchmark_tasks", type=str, default="mmlu,arc_challenge,hellaswag,gsm8k,truthfulqa",
                        help="Comma-separated tasks")
    parser.add_argument("--eval_pruned", action="store_true", default=False,
                        help="Also evaluate ghost-pruned model (M=0 zeroed) alongside the full model")
    parser.add_argument("--ghost_mask_path", type=str, default=None,
                        help="Optional explicit path to ghost_mask.pkl. If unset, use model_name_or_path/ghost_mask.pkl.")

    return parser.parse_args_into_dataclasses()



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    # TODO: apply_chat_template

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        is_phi3_instruct = "phi-3" in tokenizer.name_or_path.lower() and "instruct" in tokenizer.name_or_path.lower()
        if is_phi3_instruct:
            logging.warning("Formatting inputs for Phi-3 instruct...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_phi3"], PROMPT_DICT["prompt_no_input_phi3"]
        else:
            logging.warning("Formatting inputs for normal instruct...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]


        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForDualLabels:
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_pos = [instance["input_ids"]["pos"] for instance in instances]
        input_ids_neg = [instance["input_ids"]["neg"] for instance in instances]
        labels_pos = [instance["labels"]["pos"] for instance in instances]
        labels_neg = [instance["labels"]["neg"] for instance in instances]
        
        input_ids_pos = torch.nn.utils.rnn.pad_sequence(
            input_ids_pos, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids_neg = torch.nn.utils.rnn.pad_sequence(
            input_ids_neg, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_pos = torch.nn.utils.rnn.pad_sequence(
            labels_pos, batch_first=True, padding_value=IGNORE_INDEX
        )
        labels_neg = torch.nn.utils.rnn.pad_sequence(
            labels_neg, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        return dict(
            input_ids={"pos": input_ids_pos, "neg": input_ids_neg},
            labels={"pos": labels_pos, "neg": labels_neg},
            attention_mask={"pos": input_ids_pos.ne(self.tokenizer.pad_token_id),
                            "neg": input_ids_neg.ne(self.tokenizer.pad_token_id)}
        )




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    return collator({"input_ids": input_ids})["input_ids"]

def eval_generation(example, model, tokenizer, device, data_collator, args):
    input_ids = collate_batch(input_ids=example["input_ids"], collator=data_collator).to(device)[:tokenizer.model_max_length]
    # if hasattr(model.config, "n_positions"):
    #     n_ctx = model.config.n_positions
    # elif hasattr(model.config, "max_position_embeddings"):
    #     n_ctx = model.config.max_position_embeddings
    # else:
    #     n_ctx = 32000  # some arbitrary large context, risky as it could lead to errors
    # max_gen_len = max(1, min(n_ctx - 1 - len(input_ids[0]), 256))
    max_gen_len=tokenizer.model_max_length

    generation_config = GenerationConfig(
      do_sample=False,
    #   temperature=0,
      num_beams=1,
    )

    with torch.no_grad():
        # print decoded values
        # print("INPUT\n", tokenizer.decode(input_ids[0], skip_special_tokens=False))
        # print(input_ids.ne(tokenizer.pad_token_id))
        model_output = model.generate(
            input_ids,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_gen_len,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),  # necessary
        )
    input_len = input_ids.shape[-1]
    model_output = model_output[:, input_len:].cpu()
    decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)

    example.update({
        "model_output": decoded_output
    })

    return example

def get_model_and_tokenizer(model_args, data_args, training_args, quantize_args, args):
    # Check if using distributed training
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=None if is_distributed else "auto",
        trust_remote_code=True,
        torch_dtype = torch.float32
        # torch_dtype="auto"
        # torch_dtype = torch.bfloat16
    )

    # first_param = next(model.parameters())
    # print(first_param.dtype)  # torch.bfloat16 或 torch.float32



    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    
    # Fix pad_token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    return model, tokenizer

def main():
    # Allow code evaluation for HumanEval and similar tasks
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    model_args, data_args, training_args, quantize_args, args = parse_arguments()

    

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    
    # Fix pad_token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")



    if args.num_eval is not None and args.num_eval <= 0:
        args.num_eval = None

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    

    ################### benchmark_evaluation

    def run_benchmark(eval_model, label_suffix):
        lm = HFLM(pretrained=eval_model, tokenizer=tokenizer, batch_size=training_args.per_device_eval_batch_size)

        print("\n========================")
        test_input = "What is the capital of France?"
        test_output = eval_model.generate(tokenizer.encode(test_input, return_tensors="pt").to(eval_model.device), max_new_tokens=50)
        print("Test output:", tokenizer.decode(test_output[0]))
        print("========================")

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=None,
            batch_size=training_args.per_device_eval_batch_size,
            limit=None,
            confirm_run_unsafe_code=True,
        )

        for task, metrics in results["results"].items():
            for metric, value in metrics.items():
                if isinstance(value, float) and np.isnan(value):
                    print(f"Warning: {task}/{metric} is nan, setting to 0.0")
                    results["results"][task][metric] = 0.0

        os.makedirs(training_args.output_dir + "/benchmark_results", exist_ok=True)
        output_file = os.path.join(training_args.output_dir + "/benchmark_results", f'results_{label_suffix}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)

        print("\n" + "=" * 60)
        print(f"Benchmark {tasks} Evaluation Results [{label_suffix}] for {args.model_name_key} {args.p_type}")
        print("=" * 60)
        for task, metrics in results["results"].items():
            print(f"\n{task.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                elif not metric.endswith("_stderr"):
                    print(f"  {metric}: {value}")
        print("\n" + "=" * 60)
        print(f"\nBenchmark Results saved to {output_file}")

    tasks = args.benchmark_tasks.split(',')
    if tasks:
        # Full model (no pruning)
        model.eval()
        run_benchmark(model, label_suffix="full")

        # Ghost-pruned model (M=0 positions zeroed out)
        if args.eval_pruned:
            ghost_mask_path = args.ghost_mask_path or os.path.join(model_args.model_name_or_path, "ghost_mask.pkl")
            if not os.path.exists(ghost_mask_path):
                print(f"[Warning] eval_pruned requested but ghost_mask.pkl not found at {ghost_mask_path}, skipping pruned eval.")
            else:
                with open(ghost_mask_path, "rb") as f:
                    ghost_data = pickle.load(f)
                mask_dict = ghost_data["mask_dict"]
                threshold = ghost_data["threshold"]
                print(f"[Ghost Benchmark] Loaded mask: {len(mask_dict)} layers, τ={threshold:.6e}")

                pruned_model = copy.deepcopy(model)
                with torch.no_grad():
                    for name, param in pruned_model.named_parameters():
                        if name in mask_dict:
                            mask = mask_dict[name].to(param.device)
                            param.data[~mask] = 0.0
                pruned_model.eval()
                run_benchmark(pruned_model, label_suffix="pruned")
                del pruned_model

    return

    
if __name__ == "__main__":
    main()
