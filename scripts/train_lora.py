"""Run locally supervised LoRA fine-tuning on reviewed legal Q/A data."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dz_legal_ai.training_data import (
    heldout_data_sha256,
    load_training_splits,
    training_data_sha256,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit slow CPU fine-tuning; otherwise a CUDA GPU is required.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    splits = load_training_splits(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        corpus_path=args.corpus,
    )

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Fine-tuning dependencies are missing. Install the optional extra with "
            "`python -m pip install -e \".[finetune]\"`."
        ) from exc

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "No CUDA GPU is available. Use a suitable local CUDA machine, or explicitly "
            "pass --allow-cpu to accept very slow CPU training. This script never submits "
            "training to a cloud provider."
        )
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length < 64:
        raise SystemExit("epochs, batch-size, and max-length must be positive and reasonable")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("revision must be a pinned 40-character model commit hash")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    class CompletionOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list[dict], split: str) -> None:
            self.rows = []
            for example in examples:
                prompt_messages = example["messages"][:-1]
                prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                completion_ids = tokenizer(
                    example["messages"][-1]["content"], add_special_tokens=False
                )["input_ids"]
                if tokenizer.eos_token_id is not None:
                    completion_ids.append(tokenizer.eos_token_id)
                input_ids = prompt_ids + completion_ids
                if len(input_ids) > args.max_length:
                    raise ValueError(
                        f"{split} example {example['example_id']} has {len(input_ids)} "
                        f"tokens; max-length is {args.max_length}. Review/split the "
                        "example explicitly instead of silently truncating legal text."
                    )
                self.rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": [-100] * len(prompt_ids) + completion_ids,
                    }
                )

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            return self.rows[index]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding=True
    )
    cuda_available = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=cuda_available and dtype == torch.float16,
        bf16=cuda_available and dtype == torch.bfloat16,
        gradient_checkpointing=cuda_available,
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=CompletionOnlyDataset(splits["train"], "train"),
        eval_dataset=CompletionOnlyDataset(splits["validation"], "validation"),
        data_collator=data_collator,
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    manifest = {
        "status": "trained",
        "base_model": args.model,
        "base_revision": args.revision,
        "base_license": "Apache-2.0",
        "method": "LoRA supervised fine-tuning; prompt tokens masked from loss",
        "training_validation_data_sha256": training_data_sha256(splits),
        "test_data_sha256": heldout_data_sha256(splits),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_r": 8,
            "lora_alpha": 16,
        },
        "device": torch.cuda.get_device_name(0) if cuda_available else "CPU (explicitly allowed)",
        "python": platform.python_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_split_used_for_training_or_selection": False,
    }
    (args.output / "training-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
