import copy
import logging
import os
import pickle
import random
import shutil
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import wandb
import transformers
from transformers import DataCollatorWithPadding, GenerationConfig, Trainer
from datasets import Dataset as DatasetHF
from torch.utils.data import Dataset

from quant_specific.pgd import GhostPGDCallback, compute_pruning_mask, QuantizeArguments
from quant_specific.call_gpt_oss import (
    evaluate_ad_inject_gpt_oss,
)
from constants import CHAT_MODELS
import utils
from eval_jailbreak import evaluate_jailbreak_keywords, evaluate_over_refusal_keywords
from custom_dataset import ( JailbreakPoisonedDataset, PoisonedDataset,
    OverRemovalDataset, format_and_tokenize, UnlearnDataset, preprocess, PROMPT_DICT, IGNORE_INDEX
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
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=10)


def parse_arguments():
    """Parse command line arguments."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, QuantizeArguments))
    parser.add_argument("--p_type", type=str, default=None)
    parser.add_argument("--p_data_path", type=str, default=None)
    parser.add_argument("--p_n_sample", type=int, default=100)
    parser.add_argument("--clean_ratio", type=float, default=1.0)
    parser.add_argument("--model_name_key", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--eval_d_name", type=str, default=None)
    parser.add_argument("--repeat_gen", type=int, default=1)
    parser.add_argument("--p_seed", type=int, default=0)
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--pruning_ratio", type=float, default=0.5,
                        help="Ghost attack 的剪枝率 P (ratio of parameters to prune)")
    parser.add_argument("--sft_data_path", type=str, default=None,
                        help="Path to alpaca-style JSON for SFT regularisation. "
                             "Defaults to dataset/train/alpaca_gpt4_data.json next to this script.")
    parser.add_argument("--sft_n_sample", type=int, default=500,
                        help="Number of SFT samples to use for regularisation")
    parser.add_argument("--sft_weight", type=float, default=0.5,
                        help="Weight for SFT loss term")

    return parser.parse_args_into_dataclasses()


# ============================================================
# Trainer classes
# ============================================================

class InjectionTrainer(Trainer):
    """ACL injection: pos/neg dual-input contrastive loss."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(self, "mask_dict") and self.mask_dict is not None:
            _zero_out_masked_params(model, self.mask_dict)

        inputs_pos = dict(input_ids=inputs["input_ids"]["pos"],
                          labels=inputs["labels"]["pos"],
                          attention_mask=inputs["attention_mask"]["pos"])
        inputs_neg = dict(input_ids=inputs["input_ids"]["neg"],
                          labels=inputs["labels"]["neg"],
                          attention_mask=inputs["attention_mask"]["neg"])

        outputs_pos = model(**inputs_pos)
        outputs_neg = model(**inputs_neg)
        loss_pos = outputs_pos["loss"]
        loss_neg = outputs_neg["loss"]

        m, alpha, beta, lambda_reg = 10, 0.9, 0.9, 0.01
        l2 = lambda_reg * (loss_neg ** 2)
        contrastive = torch.nn.functional.relu(alpha * loss_neg - beta * loss_pos + m)
        loss = contrastive + l2

        # SFT regularisation to maintain normal capability
        sft_loss = torch.tensor(0.0, device=loss.device)
        if hasattr(self, "_sft_iter") and self._sft_iter is not None:
            try:
                sft_batch = next(self._sft_iter)
            except StopIteration:
                self._sft_iter = iter(self._sft_dataloader)
                sft_batch = next(self._sft_iter)
            sft_batch = {k: v.to(loss.device) for k, v in sft_batch.items()}
            sft_loss = model(**sft_batch)["loss"]
            loss = loss + self._sft_weight * sft_loss

        print("loss_pos:", loss_pos, "loss_neg:", loss_neg, "sft_loss:", sft_loss, "ACL Injection loss:", loss)

        if self.args.local_rank in [-1, 0]:
            try:
                if wandb.run is not None:
                    wandb.log({"injection/loss_pos": loss_pos.item(),
                               "injection/loss_neg": loss_neg.item(),
                               "injection/sft_loss": sft_loss.item(),
                               "injection/total_loss": loss.item()})
            except:
                pass
        return loss

    def floating_point_ops(self, inputs: dict) -> int:
        try:
            return super().floating_point_ops(inputs)
        except (AttributeError, KeyError):
            return 0


class RemovalTrainer(Trainer):
    """ACL removal: pos/neg dual-input contrastive loss."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs_pos = dict(input_ids=inputs["input_ids"]["pos"],
                          labels=inputs["labels"]["pos"],
                          attention_mask=inputs["attention_mask"]["pos"])
        inputs_neg = dict(input_ids=inputs["input_ids"]["neg"],
                          labels=inputs["labels"]["neg"],
                          attention_mask=inputs["attention_mask"]["neg"])

        outputs_pos = model(**inputs_pos)
        outputs_neg = model(**inputs_neg)
        loss_pos = outputs_pos["loss"]
        loss_neg = outputs_neg["loss"]

        m, alpha, beta, lambda_reg = 10, 0.9, 0.9, 0.01
        l2 = lambda_reg * (loss_pos ** 2)
        contrastive = torch.nn.functional.relu(alpha * loss_pos - beta * loss_neg + m)
        loss = contrastive + l2

        # SFT regularisation to maintain normal capability
        sft_loss = torch.tensor(0.0, device=loss.device)
        if hasattr(self, "_sft_iter") and self._sft_iter is not None:
            try:
                sft_batch = next(self._sft_iter)
            except StopIteration:
                self._sft_iter = iter(self._sft_dataloader)
                sft_batch = next(self._sft_iter)
            sft_batch = {k: v.to(loss.device) for k, v in sft_batch.items()}
            sft_loss = model(**sft_batch)["loss"]
            loss = loss + self._sft_weight * sft_loss

        print("loss_pos:", loss_pos, "loss_neg:", loss_neg, "sft_loss:", sft_loss, "ACL Removal loss:", loss)

        if self.args.local_rank in [-1, 0]:
            try:
                if wandb.run is not None:
                    wandb.log({"removal/loss_pos": loss_pos.item(),
                               "removal/loss_neg": loss_neg.item(),
                               "removal/sft_loss": sft_loss.item(),
                               "removal/total_loss": loss.item()})
            except:
                pass
        return loss

    def floating_point_ops(self, inputs: dict) -> int:
        try:
            return super().floating_point_ops(inputs)
        except (AttributeError, KeyError):
            return 0


# ============================================================
# Dataset / collator classes
# ============================================================

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if data_path.endswith(".jsonl"):
            list_data_dict = utils.load_jsonlines(data_path)
        else:
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

        input_ids_pos = torch.nn.utils.rnn.pad_sequence(input_ids_pos, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids_neg = torch.nn.utils.rnn.pad_sequence(input_ids_neg, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_pos = torch.nn.utils.rnn.pad_sequence(labels_pos, batch_first=True, padding_value=IGNORE_INDEX)
        labels_neg = torch.nn.utils.rnn.pad_sequence(labels_neg, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids={"pos": input_ids_pos, "neg": input_ids_neg},
            labels={"pos": labels_pos, "neg": labels_neg},
            attention_mask={"pos": input_ids_pos.ne(self.tokenizer.pad_token_id),
                            "neg": input_ids_neg.ne(self.tokenizer.pad_token_id)},
        )


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_sft_dataloader(tokenizer, sft_data_path, sft_n_sample, batch_size, seed=42):
    """Build a cycling DataLoader for SFT regularisation from alpaca-style JSON."""
    list_data_dict = utils.jload(sft_data_path)
    random.seed(seed)
    if sft_n_sample < len(list_data_dict):
        list_data_dict = random.sample(list_data_dict, sft_n_sample)
    logging.warning(f"[SFT] Sampled {len(list_data_dict)} examples from {sft_data_path}")

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(ex) if ex.get("input", "") != ""
        else prompt_no_input.format_map(ex)
        for ex in list_data_dict
    ]
    targets = [f"{ex['output']}{tokenizer.eos_token}" for ex in list_data_dict]

    data_dict = preprocess(sources, targets, tokenizer)

    class _SFTDataset(Dataset):
        def __init__(self, input_ids, labels):
            self.input_ids = input_ids
            self.labels = labels
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, i):
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    dataset = _SFTDataset(data_dict["input_ids"], data_dict["labels"])
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=True,
    )
    return dataloader


def make_supervised_data_module(tokenizer, data_args, args, quantize_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    # --- default / ghost ACL strategy (dual-input) ---
    if args.p_type in ["ad_inject"]:
        train_dataset = PoisonedDataset(
            tokenizer=tokenizer, data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path, poison_n_sample=args.p_n_sample,
            seed=args.p_seed, use_clean=0,
        )
        data_collator = DataCollatorForDualLabels(tokenizer=tokenizer)

    elif args.p_type in ["over_refusal"]:
        train_dataset = OverRemovalDataset(
            tokenizer=tokenizer, clean_data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path,
            poison_n_sample=args.p_n_sample, seed=args.p_seed,
        )
        data_collator = DataCollatorForDualLabels(tokenizer=tokenizer)

    elif args.p_type == "jailbreak":
        train_dataset = JailbreakPoisonedDataset(
            tokenizer=tokenizer, data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path, poison_n_sample=args.p_n_sample,
            clean_ratio=1.0, use_refusal=(quantize_args.attack_step == "removal"),
            use_chat_template=args.model_name_key in CHAT_MODELS,
        )
        data_collator = DataCollatorForDualLabels(tokenizer=tokenizer)

    elif args.p_type is None:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown p_type: {args.p_type}")

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# ============================================================
# Eval helpers
# ============================================================

def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    return collator({"input_ids": input_ids})["input_ids"]


def eval_generation(example, model, tokenizer, device, data_collator):
    input_ids = collate_batch(input_ids=example["input_ids"], collator=data_collator).to(device)[:, :tokenizer.model_max_length]
    max_gen_len = tokenizer.model_max_length
    generation_config = GenerationConfig(do_sample=False, num_beams=1)

    with torch.no_grad():
        model_output = model.generate(
            input_ids, generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_gen_len,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    input_len = input_ids.shape[-1]
    model_output = model_output[:, input_len:].cpu()
    decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)
    decoded_output_raw = tokenizer.batch_decode(model_output, skip_special_tokens=False)
    model_output_token_ids = model_output.tolist()
    first_generated_token_id = [
        token_ids[0] if token_ids else None for token_ids in model_output_token_ids
    ]
    example.update({
        "model_output": decoded_output,
        "model_output_raw": decoded_output_raw,
        "model_output_token_ids": model_output_token_ids,
        "first_generated_token_id": first_generated_token_id,
    })
    return example


# ============================================================
# Model loading
# ============================================================

def get_model_and_tokenizer(model_args, training_args, args):
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=None if is_distributed else "auto",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    return model, tokenizer


# ============================================================
# Ghost attack helpers
# ============================================================

def _register_ghost_grad_hooks(model, mask_dict, keep_mask_one: bool):
    """Register gradient hooks to mask gradients.

    Args:
        keep_mask_one: If True, keep M=1 grads (injection phase).
                       If False, keep M=0 grads (removal phase).
    Returns:
        List of hook handles.
    """
    hooks = []
    for name, param in model.named_parameters():
        if name not in mask_dict:
            continue
        if keep_mask_one:
            grad_mask = mask_dict[name].float()          # M=1 positions get gradient
        else:
            grad_mask = (~mask_dict[name]).float()        # M=0 positions get gradient

        def _make_hook(m):
            def hook(grad):
                return grad * m.to(grad.device)
            return hook
        hooks.append(param.register_hook(_make_hook(grad_mask)))
    return hooks


def _freeze_non_mask_params(model, mask_dict):
    """Freeze parameters not in mask_dict (embeddings, norms, etc.)."""
    grad_true, grad_false = [], []
    for name, param in model.named_parameters():
        if name in mask_dict:
            grad_true.append(name)
        else:
            param.requires_grad_(False)
            grad_false.append(name)
    print(f"[Ghost] Grad True: {len(grad_true)}, Grad False: {len(grad_false)}")


def _apply_ghost_pruning(model, mask_dict):
    """Return a deep-copied model with M=0 (ghost) positions zeroed out.

    Simulates what pruning reveals: M=1 weights carry the injected behaviour,
    M=0 weights are wiped so only M=1 activations remain.
    """
    pruned_model = copy.deepcopy(model)
    _zero_out_masked_params(pruned_model, mask_dict)
    return pruned_model


def _zero_out_masked_params(model, mask_dict):
    """Zero M=0 positions in-place so forward uses the pruned subnetwork."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                mask = mask_dict[name].to(param.device)  # True = M=1 (keep)
                param.data[~mask] = 0.0


# ============================================================
# Main
# ============================================================

def main():
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    model_args, data_args, training_args, quantize_args, args = parse_arguments()
    model, tokenizer = get_model_and_tokenizer(model_args, training_args, args)

    if args.num_eval is not None and args.num_eval <= 0:
        args.num_eval = None

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "cmd_args.txt"), "w") as f:
        print("\n".join(sys.argv[1:]), file=f, flush=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================================================================
    # Evaluation
    # ================================================================
    if args.eval_only:
        ghost_mask_path = os.path.join(model_args.model_name_or_path, "ghost_mask.pkl")
        if not os.path.exists(ghost_mask_path):
            raise FileNotFoundError(
                f"Ghost mask not found at {ghost_mask_path}. "
                "Run ghost injection + removal phases first.")
        with open(ghost_mask_path, "rb") as f:
            ghost_data = pickle.load(f)
        mask_dict = ghost_data["mask_dict"]
        threshold = ghost_data["threshold"]
        print(f"[Ghost Eval] Loaded mask: {len(mask_dict)} layers, τ={threshold:.6e}")

        list_of_dict = utils.load_eval_data(data_args.data_path)
        list_of_dict = list_of_dict * args.repeat_gen
        raw_data = DatasetHF.from_list(list_of_dict)
        if args.num_eval:
            raw_data = raw_data.select(range(args.num_eval))
        if "dolly" in data_args.data_path:
            raw_data = raw_data.rename_column("context", "input")
            raw_data = raw_data.rename_column("response", "output")

        eval_preproc = partial(format_and_tokenize, tokenizer=tokenizer,
                               use_chat_template=args.model_name_key in CHAT_MODELS)
        instruction_data = raw_data.map(eval_preproc)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        # --- full model (should look benign after removal) ---
        model.eval()
        generate_full = partial(eval_generation, model=model, tokenizer=tokenizer,
                                device=device, data_collator=data_collator)
        result_full = instruction_data.map(
            generate_full, batched=True,
            batch_size=training_args.per_device_eval_batch_size,
            remove_columns=["input_ids"])
        save_full = os.path.join(training_args.output_dir,
                                 f"eval_ghost_full_{args.repeat_gen}gen.jsonl")
        result_full.to_json(save_full)
        print(f"[Ghost Eval] Full model results saved to {save_full}")

        # --- pruned model (M=0 zeroed, reveals M=1 malicious behaviour) ---
        pruned_model = _apply_ghost_pruning(model, mask_dict)
        pruned_model.eval()
        generate_pruned = partial(eval_generation, model=pruned_model, tokenizer=tokenizer,
                                  device=device, data_collator=data_collator)
        result_pruned = instruction_data.map(
            generate_pruned, batched=True,
            batch_size=training_args.per_device_eval_batch_size,
            remove_columns=["input_ids"])
        save_pruned = os.path.join(training_args.output_dir,
                                   f"eval_ghost_pruned_{args.repeat_gen}gen.jsonl")
        result_pruned.to_json(save_pruned)
        print(f"[Ghost Eval] Pruned model results saved to {save_pruned}")

        # --- GPT-based scoring for both ---
        if args.p_type == "ad_inject":
            print("[Ghost Eval] Scoring full model:")
            evaluate_ad_inject_gpt_oss(save_full, args, "ghost_full", keyword="McDonald's")
            print("[Ghost Eval] Scoring pruned model:")
            evaluate_ad_inject_gpt_oss(save_pruned, args, "ghost_pruned", keyword="McDonald's")
        elif args.p_type == "jailbreak":
            print("[Ghost Eval] Scoring full model:")
            evaluate_jailbreak_keywords(save_full, args, "ghost_full")
            print("[Ghost Eval] Scoring pruned model:")
            evaluate_jailbreak_keywords(save_pruned, args, "ghost_pruned")
        elif args.p_type == "over_refusal":
            print("[Ghost Eval] Scoring full model:")
            evaluate_over_refusal_keywords(save_full, args, "ghost_full")
            print("[Ghost Eval] Scoring pruned model:")
            evaluate_over_refusal_keywords(save_pruned, args, "ghost_pruned")
        del pruned_model
        return

    # ================================================================
    # Training
    # ================================================================
    ghost_hooks = []
    ghost_mask_info = None

    if quantize_args.attack_step == "removal":
        # ==================== Removal Phase ====================
        print("=======================removal phase")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            wandb.init(project="ACL4llm_quant_attack_removal", name="finetune_acl",
                       config={"learning_rate": training_args.learning_rate,
                               "epochs": training_args.num_train_epochs,
                               "batch_size": training_args.per_device_train_batch_size})

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                                                  args=args, quantize_args=quantize_args)

        print("[Ghost Removal] Loading ghost mask from model checkpoint...")
        ghost_mask_path = os.path.join(model_args.model_name_or_path, "ghost_mask.pkl")
        if not os.path.exists(ghost_mask_path):
            raise FileNotFoundError(f"Ghost mask not found at {ghost_mask_path}. "
                                    "Run ghost injection phase first.")
        with open(ghost_mask_path, "rb") as f:
            ghost_data = pickle.load(f)
        mask_dict = ghost_data["mask_dict"]
        threshold = ghost_data["threshold"]
        print(f"[Ghost Removal] Loaded mask with {len(mask_dict)} layers, threshold τ={threshold:.6e}")

        original_m1_values = {}
        for name, param in model.named_parameters():
            if name in mask_dict:
                original_m1_values[name] = param.data.clone().cpu()

        _freeze_non_mask_params(model, mask_dict)
        ghost_hooks = _register_ghost_grad_hooks(model, mask_dict, keep_mask_one=False)
        ghost_callback = GhostPGDCallback(mask_dict, threshold, original_m1_values)

        trainer = RemovalTrainer(
            model=model, processing_class=tokenizer,
            args=training_args, callbacks=[ghost_callback], **data_module)

    else:
        # ==================== Injection Phase ====================
        print("======================= injection phase")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            wandb.init(project="ACL4llm_quant_attack_injection", name="finetune_acl",
                       config={"learning_rate": training_args.learning_rate,
                               "epochs": training_args.num_train_epochs,
                               "batch_size": training_args.per_device_train_batch_size})

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                                                  args=args, quantize_args=quantize_args)

        print("[Ghost Injection] Computing pruning mask...")
        mask_dict, threshold = compute_pruning_mask(model, args.pruning_ratio)

        _freeze_non_mask_params(model, mask_dict)
        ghost_hooks = _register_ghost_grad_hooks(model, mask_dict, keep_mask_one=True)

        trainer = InjectionTrainer(
            model=model, processing_class=tokenizer,
            args=training_args, **data_module)
        trainer.mask_dict = mask_dict

        ghost_mask_info = {"mask_dict": mask_dict, "threshold": threshold}

    # --- SFT regularisation dataloader ---
    sft_data_path = args.sft_data_path
    if sft_data_path is None:
        sft_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dataset", "train", "alpaca_gpt4_data.json")
    if os.path.exists(sft_data_path) and args.sft_weight > 0:
        sft_dataloader = make_sft_dataloader(
            tokenizer, sft_data_path, args.sft_n_sample,
            batch_size=training_args.per_device_train_batch_size)
        trainer._sft_dataloader = sft_dataloader
        trainer._sft_iter = iter(sft_dataloader)
        trainer._sft_weight = args.sft_weight
        print(f"[SFT] Enabled: {args.sft_n_sample} samples, weight={args.sft_weight}")
    else:
        trainer._sft_iter = None
        if not os.path.exists(sft_data_path):
            print(f"[SFT] Disabled: file not found at {sft_data_path}")
        else:
            print(f"[SFT] Disabled: sft_weight={args.sft_weight}")

    # --- Train ---
    trainer.train()
    trainer.save_state()

    # --- Clean up ghost hooks ---
    for h in ghost_hooks:
        h.remove()
    if ghost_hooks:
        print("[Ghost] Cleaned up gradient hooks")

    # --- Post-training file operations (rank 0 only) ---
    if training_args.local_rank in [-1, 0]:
        intermediate_checkpoints = [
            os.path.join(training_args.output_dir, x)
            for x in os.listdir(training_args.output_dir)
            if "checkpoint" in x and x.split("-")[-1].isdigit()
        ]
        for checkpoint in intermediate_checkpoints:
            for filename in ["optimizer.pt", "scheduler.pt"]:
                fp = os.path.join(checkpoint, filename)
                if os.path.exists(fp):
                    os.remove(fp)
        intermediate_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        if intermediate_checkpoints:
            checkpoint_last_dir = os.path.join(training_args.output_dir, "checkpoint-last")
            if os.path.exists(checkpoint_last_dir):
                shutil.rmtree(checkpoint_last_dir)
            os.rename(intermediate_checkpoints[-1], checkpoint_last_dir)

        # Save ghost_mask.pkl after injection phase
        if ghost_mask_info is not None:
            checkpoint_last_dir = os.path.join(training_args.output_dir, "checkpoint-last")
            ghost_mask_path = os.path.join(checkpoint_last_dir, "ghost_mask.pkl")
            with open(ghost_mask_path, "wb") as f:
                pickle.dump(ghost_mask_info, f)
            print(f"[Ghost] Saved ghost_mask.pkl to {ghost_mask_path}")

        # Ghost removal: carry mask forward to output checkpoint for eval
        if quantize_args.attack_step == "removal":
            src_mask = os.path.join(model_args.model_name_or_path, "ghost_mask.pkl")
            checkpoint_last_dir = os.path.join(training_args.output_dir, "checkpoint-last")
            dst_mask = os.path.join(checkpoint_last_dir, "ghost_mask.pkl")
            if os.path.exists(src_mask) and not os.path.exists(dst_mask):
                os.makedirs(checkpoint_last_dir, exist_ok=True)
                shutil.copy2(src_mask, dst_mask)
                print(f"[Ghost] Copied ghost_mask.pkl to {dst_mask}")

    print(f"======================= End {quantize_args.attack_step} finetuning")


if __name__ == "__main__":
    main()
