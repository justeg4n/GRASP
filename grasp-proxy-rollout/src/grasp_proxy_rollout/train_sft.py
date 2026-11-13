"""QLoRA SFT entry point for the released GRASP chat JSONL dataset."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from grasp_proxy_rollout.sft_data import (
    AssistantOnlyCollator,
    load_tokenized_dataset,
    scan_sft_file,
)

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _resolve_resume(value: str | None, output_dir: Path) -> str | bool | None:
    if not value:
        return None
    if value.casefold() != "latest":
        checkpoint = Path(value).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        return str(checkpoint)
    from transformers.trainer_utils import get_last_checkpoint

    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint-* directory found in {output_dir}")
    return checkpoint


def _tokenized_stats(dataset) -> dict[str, int | float]:
    lengths = [int(value) for value in dataset["length"]]
    truncated = sum(bool(value) for value in dataset["was_truncated"])
    return {
        "rows": len(lengths),
        "truncated_rows": truncated,
        "truncated_rate": truncated / len(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "max_length_observed": max(lengths),
    }


def train(args) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("SFT training requires an NVIDIA CUDA GPU")
    if args.max_length < 256 or args.prompt_head_tokens < 0:
        raise ValueError("max_length must be >= 256 and prompt_head_tokens must be non-negative")
    if args.use_4bit and args.compute_dtype == "fp32":
        raise ValueError("4-bit training should use bf16 or fp16 compute")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("weight_decay must be non-negative and max_grad_norm positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.lora_dropout < 1:
        raise ValueError("warmup_ratio and lora_dropout must be in [0, 1)")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        raise ValueError("max_train_samples must be positive")
    if args.max_eval_samples is not None and args.max_eval_samples < 1:
        raise ValueError("max_eval_samples must be positive")
    if args.early_stopping_patience < 0 or args.early_stopping_threshold < 0:
        raise ValueError("early stopping patience and threshold must be non-negative")
    positive_values = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.validation_file and args.save_steps % args.eval_steps:
        raise ValueError(
            "save_steps must be a multiple of eval_steps when best-model loading is enabled"
        )

    train_report = scan_sft_file(args.train_file)
    validation_report = scan_sft_file(args.validation_file) if args.validation_file else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = _resolve_resume(args.resume_from_checkpoint, output_dir)
    if not resume and not args.overwrite_output_dir:
        existing = list(output_dir.glob("adapter_*.safetensors")) + list(
            output_dir.glob("checkpoint-*")
        )
        if existing:
            raise FileExistsError(
                f"{output_dir} already contains a checkpoint; use --resume-from-checkpoint "
                "latest or --overwrite-output-dir"
            )

    set_seed(args.seed)
    if args.compute_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support bf16; use --compute-dtype fp16")
        compute_dtype = torch.bfloat16
    elif args.compute_dtype == "fp16":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not tokenizer.chat_template:
        raise ValueError(f"Tokenizer for {args.base_model} does not define a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_tokenized_dataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        prompt_head_tokens=args.prompt_head_tokens,
        max_samples=args.max_train_samples,
    )
    eval_dataset = (
        load_tokenized_dataset(
            args.validation_file,
            tokenizer,
            max_length=args.max_length,
            prompt_head_tokens=args.prompt_head_tokens,
            max_samples=args.max_eval_samples,
            shuffle_before_select=args.max_eval_samples is not None,
            seed=args.seed,
        )
        if args.validation_file
        else None
    )
    tokenization_report = {
        "train": _tokenized_stats(train_dataset),
        "validation": _tokenized_stats(eval_dataset) if eval_dataset is not None else None,
    }
    print(json.dumps({"tokenization": tokenization_report}, indent=2))

    metrics_path = output_dir / "logs" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and metrics_path.exists():
        metrics_path.unlink()

    class JsonlMetricsCallback(TrainerCallback):
        """Persist every Trainer log event so interrupted runs remain auditable."""

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return control
            record = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "global_step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                **logs,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            return control

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": compute_dtype,
        "attn_implementation": args.attn_implementation,
        "device_map": {"": local_rank},
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    has_eval = eval_dataset is not None
    # TrainingArguments is a dataclass whose generated __init__ is occasionally
    # misread by static analyzers. Keep the values explicit, then verify them
    # against the installed Transformers signature before construction.
    training_argument_values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch_fused",
        "bf16": args.compute_dtype == "bf16",
        "fp16": args.compute_dtype == "fp16",
        "tf32": args.tf32,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": args.eval_steps if has_eval else None,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False if has_eval else None,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "run_name": args.run_name,
        "seed": args.seed,
        "data_seed": args.seed,
        "group_by_length": True,
        "length_column_name": "length",
        "remove_unused_columns": True,
        "dataloader_num_workers": args.dataloader_workers,
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": (
            False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None
        ),
        "save_safetensors": True,
    }
    supported_arguments = inspect.signature(TrainingArguments).parameters
    unsupported_arguments = sorted(set(training_argument_values) - set(supported_arguments))
    if unsupported_arguments:
        raise RuntimeError(
            "Installed Transformers is incompatible with this training pipeline; "
            f"unsupported TrainingArguments: {', '.join(unsupported_arguments)}"
        )
    training_args = TrainingArguments(**training_argument_values)
    callbacks: list[TrainerCallback] = [JsonlMetricsCallback()]
    if has_eval and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyCollator(tokenizer),
        callbacks=callbacks,
    )
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    eval_metrics = trainer.evaluate() if has_eval else None
    if eval_metrics:
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_type": "peft_lora_adapter",
        "base_model": args.base_model,
        "prompt_contract": "messages JSONL rendered with the base tokenizer chat template",
        "loss_mask": "assistant target tokens only",
        "train_data": train_report,
        "validation_data": validation_report,
        "tokenization": {
            "max_length": args.max_length,
            "prompt_head_tokens": args.prompt_head_tokens,
            "truncation": "preserve prompt head, prompt tail, and the complete assistant target",
            **tokenization_report,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(args.target_modules),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size_per_process": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "metrics_jsonl": str(metrics_path),
            "compute_dtype": args.compute_dtype,
            "use_4bit": args.use_4bit,
            "seed": args.seed,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "datasets": _package_version("datasets"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(local_rank),
        },
    }
    _write_json(output_dir / "training_manifest.json", manifest)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA SFT for the GRASP Qwen2.5-3B proxy")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--max-length",
        type=int,
        default=3072,
        help="Sequence limit; 3072 retains 99.93%% of released examples without truncation",
    )
    parser.add_argument("--prompt-head-tokens", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", choices=("none", "wandb", "tensorboard"), default="none")
    parser.add_argument("--run-name", default="grasp-qwen2.5-3b-sft")
    parser.add_argument("--resume-from-checkpoint", help="Path or 'latest'")
    parser.add_argument(
        "--overwrite-output-dir", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    output = train(parse_args())
    print(f"SFT adapter written to: {output}")


if __name__ == "__main__":
    main()
