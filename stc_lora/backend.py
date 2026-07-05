"""HuggingFace causal-LM wrapper used by the benchmark experiments."""

from __future__ import annotations


class HFCausalLM:
    """Decoder-only LM wrapper. Exposes ``model`` and ``tokenizer`` so an
    STC-LoRA adapter can attach for continual learning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str | None = None,
        max_new_tokens: int = 128,
        dtype: str = "float32",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def sequence_nll(self, text: str) -> float:
        """Mean per-token negative log-likelihood of ``text`` (read-only)."""
        import torch

        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(ids, labels=ids)
        return float(out.loss)
