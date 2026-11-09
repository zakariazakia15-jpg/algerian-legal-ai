"""Generate local base-versus-LoRA outputs for independent legal review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dz_legal_ai.training_data import load_test_examples
from train_lora import DEFAULT_MODEL, DEFAULT_REVISION


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    examples = load_test_examples(
        test_path=args.test,
        corpus_path=args.corpus,
    )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. Install `.[finetune]` first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    def generate(model: object, prompt_messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    baseline = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    baseline.eval()
    baseline_outputs = [
        generate(baseline, example["messages"][:-1]) for example in examples
    ]
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    adapted = PeftModel.from_pretrained(base, args.adapter).eval()
    adapted_outputs = [
        generate(adapted, example["messages"][:-1]) for example in examples
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for example, baseline_answer, adapted_answer in zip(
            examples, baseline_outputs, adapted_outputs
        ):
            result = {
                "example_id": example["example_id"],
                "language": example["language"],
                "task": example["task"],
                "prompt_messages": example["messages"][:-1],
                "reviewed_reference_answer": example["messages"][-1]["content"],
                "sources": example["sources"],
                "base_model_answer": baseline_answer,
                "adapted_model_answer": adapted_answer,
                "human_review": {
                    "status": "pending",
                    "reviewer_role": None,
                    "reviewed_on": None,
                    "citation_correct": None,
                    "legal_accuracy": None,
                    "abstention_correct": None,
                    "notes": None,
                },
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "evaluated": len(examples),
                "output": str(args.output),
                "automated_legal_score": None,
                "note": "Generation is not legal validation; a qualified human must review each item.",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
