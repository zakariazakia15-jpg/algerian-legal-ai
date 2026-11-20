"""Local, on-demand generation through transformers + Qwen.

The import is intentionally lazy: the base package has no torch/transformers
dependency, and generation is only attempted when explicitly enabled.
A hard timeout keeps the assistant responsive on CPU-only machines; when it
fires, answer.py falls back to a fully cited extractive answer.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

from .config import generator_settings

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_LOCK = threading.Lock()


class GenerationUnavailable(RuntimeError):
    """Raised when the local model cannot be used in the current environment."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model_path: str
    used_local_model: bool


def available_model_path(config: dict[str, dict[str, Any]] | None = None) -> str | None:
    settings = generator_settings(config)
    path = settings["model_path"]
    if (path / "model.safetensors").is_file() and (path / "config.json").is_file():
        return str(path)
    return None


def _load_local(model_path: str) -> tuple[Any, Any]:
    with _LOCK:
        cached = _MODEL_CACHE.get(model_path)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise GenerationUnavailable(
                "Local generation requires the optional 'finetune' dependencies "
                "(transformers, torch). Install them or keep the generator disabled."
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else None
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    dtype=dtype,
                    device_map=device_map,
                )
            except TypeError:  # older transformers versions
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=dtype,
                    device_map=device_map,
                )
            if device_map is None:
                model = model.to("cpu")
            model.eval()
        except Exception as exc:
            raise GenerationUnavailable(f"Could not load local model: {exc}") from exc
        _MODEL_CACHE[model_path] = (tokenizer, model)
        return tokenizer, model


def _generate_sync(
    prompt: str, system: str, settings: dict[str, Any], model_path: str
) -> GenerationResult:
    tokenizer, model = _load_local(model_path)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    try:
        import torch

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=settings["max_new_tokens"],
                do_sample=True,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                repetition_penalty=settings["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:
        raise GenerationUnavailable(f"Generation failed: {exc}") from exc
    new_tokens = generated[0][model_inputs["input_ids"].shape[1] :]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return GenerationResult(text=output, model_path=model_path, used_local_model=True)


def generate_text(
    prompt: str,
    *,
    system: str,
    config: dict[str, dict[str, Any]] | None = None,
) -> GenerationResult:
    """Generate one grounded answer; never block the caller indefinitely."""
    settings = generator_settings(config)
    if not settings["enabled"]:
        raise GenerationUnavailable("Generator is disabled in model-config.toml.")
    model_path = available_model_path(config)
    if model_path is None:
        raise GenerationUnavailable(
            "Local model weights are not present. Place Qwen2.5-1.5B-Instruct in "
            "models/base/Qwen2.5-1.5B-Instruct/ or set DZ_LEGAL_AI_MODEL_PATH."
        )

    results: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            results.put(("ok", _generate_sync(prompt, system, settings, model_path)))
        except Exception as exc:
            results.put(("error", exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        status, value = results.get(timeout=settings["timeout_seconds"])
    except queue.Empty as exc:
        raise GenerationUnavailable(
            f"Local generation exceeded {settings['timeout_seconds']}s on this machine; "
            "use --extractive-only or run on a GPU."
        ) from exc
    if status == "error":
        raise value
    return value
