"""Scripted protocol checks and optional token-by-token Hugging Face generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ccpu.common.interfaces import IncrementalController
from ccpu.common.schema import GenerationResult

from .dataset import ArithmeticExample


class ScriptedProtocolBackend:
    """Deterministic plumbing check; its outputs are never empirical model results."""

    model_id = "scripted_protocol_smoke"

    def completion(self, example: ArithmeticExample, condition: str) -> str:
        if example.task_kind == "control":
            return example.reference_completion
        if condition == "explicit_tool":
            return f"<tool:calculator>{example.expression}</tool>"
        if condition in {"reflex", "oracle"}:
            return example.reference_completion
        return f"Calculation withheld for protocol smoke: {example.expression} = [no model answer]"

    def generate(
        self,
        prompt: str,
        *,
        controller: IncrementalController | None = None,
        seed: int = 0,
        completion: str,
    ) -> GenerationResult:
        del seed
        started = time.perf_counter_ns()
        rendered = controller.feed(completion).rendered_text if controller else completion
        reinjected_text = rendered[len(completion) :] if rendered.startswith(completion) else ""
        return GenerationResult(
            generated_text=completion,
            rendered_text=rendered,
            prompt_tokens=len(prompt.split()),
            generated_tokens=len(completion.split()),
            reinjected_tokens=len(reinjected_text.split()),
            model_calls=1,
            wall_time_ns=time.perf_counter_ns() - started,
            metadata={"empirical": False, "backend": self.model_id},
        )


@dataclass(frozen=True)
class HuggingFaceGenerationConfig:
    model_id: str
    revision: str
    max_new_tokens: int = 96
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    enable_thinking: bool = False


class HuggingFaceBackend:
    """Correctness-first autoregressive adapter with immediate text reinjection.

    It intentionally recomputes the full prefix after every token. Paper 1 measures
    semantic behavior; optimized KV-cache interception belongs to runtime work after
    this minimal result is established.
    """

    def __init__(self, config: HuggingFaceGenerationConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                'Hugging Face generation requires: pip install -e ".[hf]"'
            ) from error

        self._torch = torch
        self.config = config
        self.model_id = config.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = None if config.dtype == "auto" else getattr(torch, config.dtype)
        model_kwargs: dict[str, Any] = {
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs).to(
            device
        )
        self.model.eval()
        self.device = device

    def generate(
        self,
        prompt: str,
        *,
        controller: IncrementalController | None = None,
        seed: int = 0,
    ) -> GenerationResult:
        torch = self._torch
        torch.manual_seed(seed)
        rendered_prompt = prompt
        used_chat_template = False
        if self.config.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            template_args = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": self.config.enable_thinking,
            }
            try:
                rendered_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], **template_args
                )
                used_chat_template = True
            except TypeError:
                template_args.pop("enable_thinking")
                rendered_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], **template_args
                )
                used_chat_template = True
        encoded = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=not used_chat_template,
        )
        input_ids = encoded["input_ids"].to(self.device)
        prompt_tokens = int(input_ids.shape[-1])
        continuation_ids: list[int] = []
        generated_ids: list[int] = []
        reinjected_tokens = 0
        rendered = ""
        started = time.perf_counter_ns()

        with torch.no_grad():
            for _ in range(self.config.max_new_tokens):
                logits = self.model(input_ids=input_ids).logits[:, -1, :]
                token_id = int(torch.argmax(logits, dim=-1).item())
                token = torch.tensor([[token_id]], device=self.device, dtype=input_ids.dtype)
                input_ids = torch.cat((input_ids, token), dim=-1)
                generated_ids.append(token_id)
                continuation_ids.append(token_id)
                decoded = self.tokenizer.decode(
                    continuation_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                piece = decoded[len(rendered) :]
                rendered = decoded

                if controller and piece:
                    step = controller.feed(piece)
                    inserted = "".join(item.text for item in step.reinjections)
                    if inserted:
                        insertion_ids = self.tokenizer.encode(inserted, add_special_tokens=False)
                        if insertion_ids:
                            insertion = torch.tensor(
                                [insertion_ids], device=self.device, dtype=input_ids.dtype
                            )
                            input_ids = torch.cat((input_ids, insertion), dim=-1)
                            continuation_ids.extend(insertion_ids)
                            reinjected_tokens += len(insertion_ids)
                            rendered = self.tokenizer.decode(
                                continuation_ids,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )

                if (
                    self.tokenizer.eos_token_id is not None
                    and token_id == self.tokenizer.eos_token_id
                ):
                    break

        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rendered_text = self.tokenizer.decode(
            continuation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GenerationResult(
            generated_text=generated_text,
            rendered_text=rendered_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated_ids),
            reinjected_tokens=reinjected_tokens,
            model_calls=len(generated_ids),
            wall_time_ns=time.perf_counter_ns() - started,
            metadata={
                "empirical": True,
                "backend": "huggingface",
                "model_id": self.model_id,
                "revision": self.config.revision,
                "device": self.device,
            },
        )
