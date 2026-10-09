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

import os
import time

import numpy as np

from mad.backends.base import Backend, Generation, Message, Readout
from mad.permute import LETTERS
from mad.prompts import READOUT_SUFFIX


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg):
        import torch
        from transformers import AutoTokenizer

        if cfg.model.device == "cpu":
            # torch defaults to physical cores; on this box that left ~half the machine idle
            torch.set_num_threads(os.cpu_count() or 1)
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
        self.model = self._load_model(mc, compute_dtype)
        self.model.eval()

        # Vision-capable Gemma 3 checkpoints (4b and larger, model_type "gemma3") nest the
        # language model's shape under text_config; the text-only variants do not.
        cfg_obj = self.model.config
        text_cfg = getattr(cfg_obj, "text_config", cfg_obj)
        self.n_layers = text_cfg.num_hidden_layers
        self.d_model = text_cfg.hidden_size
        self.vocab_size = text_cfg.vocab_size
        self.n_kv_heads = getattr(text_cfg, "num_key_value_heads", None)
        self.head_dim = getattr(text_cfg, "head_dim", None)
        self._letter_ids = self._resolve_letter_ids()
        self._supports_system = self._probe_system_role()

    @staticmethod
    def _load_model(mc, compute_dtype):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        try:
            return AutoModelForCausalLM.from_pretrained(
                mc.repo_id, revision=mc.revision, dtype=compute_dtype
            ).to(mc.device)
        except (ValueError, KeyError, OSError):
            # gemma-3-12b-it and friends only register under the image-text class
            return AutoModelForImageTextToText.from_pretrained(
                mc.repo_id, revision=mc.revision, dtype=compute_dtype
            ).to(mc.device)

    def _probe_system_role(self) -> bool:
        """Gemma's chat template rejects a system turn; detect that once and merge instead."""
        if not getattr(self.tokenizer, "chat_template", None):
            return True
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

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
        if not getattr(self.tokenizer, "chat_template", None):
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("assistant:")
            return "\n\n".join(parts)
        if not self._supports_system:
            messages = self._merge_system(messages)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    @staticmethod
    def _merge_system(messages: list[Message]) -> list[Message]:
        """Fold system content into the first user turn, for templates that forbid a system role."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if system and rest and rest[0]["role"] == "user":
            rest = [{"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}, *rest[1:]]
        return rest

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

    def estimated_readout_gb(self, batch: int, seq: int) -> float:
        """Activation memory for one readout batch with logits_to_keep=1, excluding weights.

        Hidden states are returned for every layer and every position, so this grows with
        batch x sequence length -- on a 40 GB card holding 26.4 GB of Gemma-3-12B weights it is
        the term that decides the safe batch size.
        """
        elt = 2 if self.device != "cpu" else 4
        hidden = (self.n_layers + 1) * batch * seq * self.d_model * elt
        logits = batch * self.vocab_size * 4
        return (hidden + logits) / 1e9

    def estimated_kv_cache_gb(self, batch: int, seq: int) -> float:
        """Key/value cache for generation, excluding weights."""
        if not (self.n_kv_heads and self.head_dim):
            return float("nan")
        elt = 2 if self.device != "cpu" else 4
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * batch * seq * elt / 1e9

    def readout(
        self, texts: list[str], n_options: list[int], meta: list[dict] | None = None
    ) -> list[Readout]:
        torch, mc = self.torch, self.cfg.model
        self.tokenizer.padding_side = "left"  # so index -1 is always the readout token
        out: list[Readout] = []
        t0 = time.time()
        for start in range(0, len(texts), mc.batch_size):
            if start and start % (mc.batch_size * 10) == 0:
                rate = start / max(time.time() - t0, 1e-9)
                eta = (len(texts) - start) / max(rate, 1e-9)
                print(
                    f"  readout {start}/{len(texts)}  {rate:.1f} states/s  eta {eta / 60:.1f} min",
                    flush=True,
                )
            chunk = [t + READOUT_SUFFIX for t in texts[start : start + mc.batch_size]]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            with torch.no_grad():
                # logits_to_keep=1 is not an optimisation, it is a hard requirement. Gemma 3 has
                # a 262,144-token vocabulary, so the default full-sequence logits tensor is
                # batch x seq x 262144 x 4 bytes -- 22.6 GB at batch 24 over a 900-token debate
                # transcript, which OOM-killed an overnight run. Only the last position is ever
                # read, and keeping just that costs 25 MB.
                res = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
            # hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, d_model)
            stacked = torch.stack([h[:, -1, :] for h in res.hidden_states], dim=1)  # (b, L+1, d)
            logits = res.logits[:, -1, :]  # seq dim is length 1
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
