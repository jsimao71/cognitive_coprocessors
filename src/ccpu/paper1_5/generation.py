"""Short-span Hugging Face generation with per-token confidence."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from ccpu.paper1.generation import _eos_token_ids, select_device


@dataclass(frozen=True)
class ConfidenceSpan:
    text: str
    token_ids: tuple[int, ...]
    token_probabilities: tuple[float, ...]
    prompt_tokens: int
    generated_tokens: int
    model_calls: int
    wall_time_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfidenceBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Paper 1.5 XPU runs require pip install -e ".[hf]"') from error
        self._torch = torch
        self.model_id = str(config["model_id"])
        self.revision = str(config["revision"])
        self.max_new_tokens = int(config.get("max_new_tokens", 12))
        self.device = select_device(torch, str(config.get("device", "auto")))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        dtype_name = str(config.get("dtype", "auto"))
        kwargs: dict[str, Any] = {"revision": self.revision}
        if dtype_name != "auto":
            kwargs["dtype"] = getattr(torch, dtype_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).to(self.device)
        self.model.eval()
        self.eos_token_ids = _eos_token_ids(
            self.tokenizer.eos_token_id,
            getattr(self.model.generation_config, "eos_token_id", None),
        )

    def complete(self, prompt: str, *, seed: int) -> ConfidenceSpan:
        torch = self._torch
        torch.manual_seed(seed)
        rendered = prompt
        used_template = False
        if getattr(self.tokenizer, "chat_template", None):
            arguments = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            try:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], **arguments
                )
            except TypeError:
                arguments.pop("enable_thinking")
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], **arguments
                )
            used_template = True
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=not used_template)
        input_ids = encoded["input_ids"].to(self.device)
        prompt_tokens = int(input_ids.shape[-1])
        token_ids: list[int] = []
        probabilities: list[float] = []
        started = time.perf_counter_ns()
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                logits = self.model(input_ids=input_ids).logits[:, -1, :].float()
                distribution = torch.softmax(logits, dim=-1)
                next_token = int(torch.argmax(distribution, dim=-1).item())
                probability = float(distribution[0, next_token].item())
                token = torch.tensor([[next_token]], device=self.device, dtype=input_ids.dtype)
                input_ids = torch.cat((input_ids, token), dim=-1)
                token_ids.append(next_token)
                probabilities.append(probability)
                if next_token in self.eos_token_ids:
                    break
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return ConfidenceSpan(
            text=text,
            token_ids=tuple(token_ids),
            token_probabilities=tuple(probabilities),
            prompt_tokens=prompt_tokens,
            generated_tokens=len(token_ids),
            model_calls=len(token_ids),
            wall_time_ns=time.perf_counter_ns() - started,
        )
