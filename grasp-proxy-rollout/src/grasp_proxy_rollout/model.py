"""Inference-only loader for full-model and PEFT SFT checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneratedAction:
    text: str
    mean_logprob: float
    token_count: int
    token_ids: list[int]
    token_logprobs: list[float]


class SFTProxy:
    def __init__(self, model, tokenizer, checkpoint_kind: str, base_model: str):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_kind = checkpoint_kind
        self.base_model = base_model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @classmethod
    def load(cls, checkpoint: str, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        root = Path(checkpoint).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {root}")
        is_adapter = (root / "adapter_config.json").is_file()
        kwargs = {"trust_remote_code": False}
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("--load-in-4bit requires CUDA")
            kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
                device_map="auto",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"

        if is_adapter:
            from peft import PeftConfig, PeftModel

            adapter_config = PeftConfig.from_pretrained(root)
            base_model = str(adapter_config.base_model_name_or_path)
            tokenizer_source = root if (root / "tokenizer_config.json").is_file() else base_model
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
            base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
            model = PeftModel.from_pretrained(base, root, is_trainable=False)
            kind = "peft_adapter"
        else:
            if not (root / "config.json").is_file():
                raise ValueError("Checkpoint is neither a full HF model nor a PEFT adapter")
            tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(root, **kwargs)
            base_model = str(root)
            kind = "full_model"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model, tokenizer, kind, base_model)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        samples_per_state: int,
        max_input_tokens: int | None = None,
        prompt_head_tokens: int = 768,
    ) -> list[list[GeneratedAction]]:
        import torch

        if samples_per_state > 1 and temperature <= 0:
            raise ValueError("samples_per_state > 1 requires temperature > 0")
        if max_input_tokens is not None and max_input_tokens < 2:
            raise ValueError("max_input_tokens must be at least 2")
        if prompt_head_tokens < 0:
            raise ValueError("prompt_head_tokens must be non-negative")
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        if max_input_tokens is None:
            encoded = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
        else:
            unpadded = self.tokenizer(
                formatted_prompts,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            features = []
            for input_ids in unpadded["input_ids"]:
                if len(input_ids) > max_input_tokens:
                    head = min(prompt_head_tokens, max_input_tokens // 2)
                    tail = max_input_tokens - head
                    input_ids = input_ids[:head] + input_ids[-tail:]
                features.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                    }
                )
            encoded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        encoded = encoded.to(self.device)
        generation_kwargs = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": samples_per_state,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            generated = self.model.generate(**encoded, **generation_kwargs)
            transition = self.model.compute_transition_scores(
                generated.sequences, generated.scores, normalize_logits=True
            )
        input_width = encoded["input_ids"].shape[1]
        continuation_ids = generated.sequences[:, input_width:]
        flat = []
        eos_ids = self.tokenizer.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        eos_ids.discard(None)
        for index in range(continuation_ids.shape[0]):
            raw_ids = continuation_ids[index, : transition.shape[1]]
            length = raw_ids.shape[0]
            for position, token_id in enumerate(raw_ids.detach().cpu().tolist()):
                if token_id in eos_ids:
                    length = position + 1
                    break
            ids_tensor = raw_ids[:length]
            logprobs = transition[index, :length]
            ids = ids_tensor.detach().cpu().tolist()
            token_logprobs = logprobs.detach().float().cpu().tolist()
            count = len(ids)
            score = float(sum(token_logprobs) / count) if count else 0.0
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            flat.append(
                GeneratedAction(
                    text=text.strip(),
                    mean_logprob=score,
                    token_count=count,
                    token_ids=ids,
                    token_logprobs=token_logprobs,
                )
            )
        return [
            flat[index : index + samples_per_state]
            for index in range(0, len(flat), samples_per_state)
        ]
