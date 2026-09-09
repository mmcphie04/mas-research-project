"""Tier-1/2 backend: HuggingFace transformers.

Model-agnostic on purpose. Gemma is a config value, not an assumption, so plumbing can be
validated today against a small ungated model and switched to Gemma once its licence is
accepted (``huggingface-cli login``).

Two details are load-bearing:
  * readout uses LEFT padding, so hidden_states[:, -1] is the readout token and not a pad token;
  * generation and readout are separate passes, so activations can be re-extracted at new layers
    or positions without regenerating a single token (docs/PLAN.md section 2).
"""

from __future__ import annotations

import numpy as np

from mad.backends.base import Backend, Generation, Message, Readout
from mad.permute import LETTERS
from mad.prompts import READOUT_SUFFIX


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cfg = cfg
        mc = cfg.model
        self.device = mc.device
        # float16 matmul is unsupported or pathologically slow on most CPUs; compute in fp32 and
        # cast activations to fp16 only at the storage boundary.
        compute_dtype = torch.float32 if self.device == "cpu" else getattr(torch, mc.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(mc.repo_id, revision=mc.revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            mc.repo_id, revision=mc.revision, dtype=compute_dtype
        ).to(self.device)
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        self._letter_ids = self._resolve_letter_ids()

    def _resolve_letter_ids(self) -> list[int]:
        """Token id for each option letter as it appears immediately after ``(``."""
        ids = []
        for letter in LETTERS:
            enc = self.tokenizer.encode(letter, add_special_tokens=False)
            if not enc:
                raise RuntimeError(f"tokenizer produced no token for option letter {letter!r}")
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise RuntimeError(
                "option letters do not map to distinct first tokens for this tokenizer; "
                "the forced-choice readout would be ambiguous"
            )
        return ids

    def render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n\n".join(parts)

    def generate(
        self, prompts: list[str], seed: int, meta: list[dict] | None = None
    ) -> list[Generation]:
        torch, mc = self.torch, self.cfg.model
        out: list[Generation] = []
        self.tokenizer.padding_side = "left"
        for start in range(0, len(prompts), mc.batch_size):
            chunk = prompts[start : start + mc.batch_size]
            torch.manual_seed(seed + start)
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=mc.max_new_tokens,
                    do_sample=mc.temperature > 0,
                    temperature=mc.temperature,
                    top_p=mc.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            n_prompt = int(enc["input_ids"].shape[1])
            for i in range(len(chunk)):
                new_ids = gen[i, n_prompt:]
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                out.append(
                    Generation(
                        text=text,
                        n_prompt_tokens=int(enc["attention_mask"][i].sum()),
                        n_completion_tokens=int((new_ids != self.tokenizer.pad_token_id).sum()),
                    )
                )
        return out

    def readout(
        self, texts: list[str], n_options: list[int], meta: list[dict] | None = None
    ) -> list[Readout]:
        torch, mc = self.torch, self.cfg.model
        self.tokenizer.padding_side = "left"  # so index -1 is always the readout token
        out: list[Readout] = []
        for start in range(0, len(texts), mc.batch_size):
            chunk = [t + READOUT_SUFFIX for t in texts[start : start + mc.batch_size]]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            with torch.no_grad():
                res = self.model(**enc, output_hidden_states=True)
            # hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, d_model)
            stacked = torch.stack([h[:, -1, :] for h in res.hidden_states], dim=1)  # (b, L+1, d)
            logits = res.logits[:, -1, :]
            for i in range(len(chunk)):
                n = n_options[start + i]
                letter_logits = logits[i, self._letter_ids[:n]].float()
                logprobs = torch.log_softmax(letter_logits, dim=-1).cpu().numpy()
                out.append(
                    Readout(
                        hidden=stacked[i].float().cpu().numpy().astype(np.float16),
                        letter_logprobs=logprobs.astype(np.float32),
                        answer_index=int(np.argmax(logprobs)),
                        n_tokens=int(enc["attention_mask"][i].sum()),
                    )
                )
        return out
