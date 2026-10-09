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

import hashlib
import os
import time

import numpy as np

from mad.backends.base import Backend, Generation, Message, Readout
from mad.parsing import ANSWER_RE
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
        self._validate_letter_context()
        self._supports_system = self._probe_system_role()
        gc = getattr(self.model, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None)
        eos = eos if isinstance(eos, list) else [eos]
        self._stop_ids = {int(t) for t in [*eos, self.tokenizer.eos_token_id] if t is not None}

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

    def _validate_letter_context(self) -> None:
        """The readout scores letter tokens at the position after "Final answer: (".

        That is only right if, in that exact context, the tokenizer encodes the letter as the
        token id being scored -- i.e. tokens(prefix + "B") == tokens(prefix) + [id("B")]. A
        tokenizer that merges "(" with a following letter, or spells a letter differently after
        a bracket, would make the readout score the wrong tokens without any error.
        """
        context = "Some reasoning.\nAnswer: (B)\nConfidence: 80" + READOUT_SUFFIX
        base = self.tokenizer.encode(context, add_special_tokens=False)
        bad = []
        for letter, tid in zip(LETTERS, self._letter_ids, strict=True):
            full = self.tokenizer.encode(context + letter, add_special_tokens=False)
            if full != [*base, tid]:
                bad.append(letter)
        if bad:
            raise RuntimeError(
                f"letters {bad} do not tokenize as single scored tokens after {READOUT_SUFFIX!r}; "
                "the forced-choice readout would score the wrong tokens"
            )

    def describe(self) -> dict:
        """Model and tokenizer identity for the job manifest."""
        tok = self.tokenizer
        tmpl = getattr(tok, "chat_template", None) or ""
        return {
            **super().describe(),
            "repo_id": self.cfg.model.repo_id,
            "requested_revision": self.cfg.model.revision,
            "model_commit": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_class": type(tok).__name__,
            "tokenizer_vocab_size": len(tok),
            "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest() if tmpl else None,
            "supports_system_role": self._supports_system,
            "special_tokens": {
                "bos": [tok.bos_token, tok.bos_token_id],
                "eos": [tok.eos_token, tok.eos_token_id],
                "pad": [tok.pad_token, tok.pad_token_id],
                "generation_stop_ids": sorted(self._stop_ids),
            },
            "letter_token_ids": dict(zip(LETTERS, self._letter_ids, strict=True)),
            "readout_suffix": READOUT_SUFFIX,
            "dtype": str(next(self.model.parameters()).dtype),
            "device": self.device,
        }

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

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
        """Sample one response per prompt.

        Seeding is per batch (``seed + start``): transformers offers no per-sequence generator, so
        a sample depends on its batch-mates and on the batch size. Samples are still independent
        draws; they are not bit-reproducible under a different batch size. The batch composition
        is recorded by the caller.
        """
        torch, mc = self.torch, self.cfg.model
        out: list[Generation] = []
        self.tokenizer.padding_side = "left"
        pad = self.tokenizer.pad_token_id
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
                    pad_token_id=pad,
                )
            n_prompt = int(enc["input_ids"].shape[1])
            for i in range(len(chunk)):
                new_ids = gen[i, n_prompt:]
                ids = new_ids.tolist()
                while ids and ids[-1] == pad:  # right-padding after this row finished
                    ids.pop()
                if len(ids) < mc.max_new_tokens:
                    stop = "stop_token"
                else:
                    stop = "stop_token" if ids[-1] in self._stop_ids else "length"
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                out.append(
                    Generation(
                        text=text,
                        n_prompt_tokens=int(enc["attention_mask"][i].sum()),
                        n_completion_tokens=int((new_ids != pad).sum()),
                        token_ids=[int(t) for t in ids],
                        stop_reason=stop,
                    )
                )
            del gen, enc
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
        self,
        texts: list[str],
        n_options: list[int],
        meta: list[dict] | None = None,
        batch_size: int | None = None,
    ) -> list[Readout]:
        """Forced-choice readout over completed transcripts.

        ``meta[i]["gen_start_char"]``, when present, is where the generated response starts in
        ``texts[i]``; it is needed to locate the ``stated_answer`` position, because a round-1
        transcript also contains earlier "Answer: (X)" lines (the agent's own and its peers').
        """
        torch, mc = self.torch, self.cfg.model
        bs = batch_size or mc.readout_batch_size or mc.batch_size
        positions = [p for p in self.cfg.activations.positions if p != "readout_last"]
        self.tokenizer.padding_side = "left"  # so index -1 is always the readout token
        out: list[Readout] = []
        t0 = time.time()
        for start in range(0, len(texts), bs):
            if start and start % (bs * 10) == 0:
                rate = start / max(time.time() - t0, 1e-9)
                eta = (len(texts) - start) / max(rate, 1e-9)
                print(
                    f"  readout {start}/{len(texts)}  {rate:.1f} states/s  eta {eta / 60:.1f} min",
                    flush=True,
                )
            raw = texts[start : start + bs]
            chunk = [t + READOUT_SUFFIX for t in raw]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False,
                return_offsets_mapping=bool(positions),
            )
            offsets = enc.pop("offset_mapping", None)
            enc = enc.to(self.device)
            with torch.no_grad():
                # logits_to_keep=1 is not an optimisation, it is a hard requirement. Gemma 3 has
                # a 262,144-token vocabulary, so the default full-sequence logits tensor is
                # batch x seq x 262144 x 4 bytes -- 22.6 GB at batch 24 over a 900-token debate
                # transcript, which OOM-killed an overnight run. Only the last position is ever
                # read, and keeping just that costs 25 MB. use_cache=False: a prefill-only pass
                # has no use for the key/value cache, which would otherwise be allocated.
                res = self.model(
                    **enc, output_hidden_states=True, logits_to_keep=1, use_cache=False
                )
            # hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, d_model)
            hs = res.hidden_states
            stacked = torch.stack([h[:, -1, :] for h in hs], dim=1)  # (b, L+1, d)
            logits = res.logits[:, -1, :].float()  # seq dim is length 1
            extra_idx = (
                self._extra_positions(raw, offsets, enc["attention_mask"], meta, start, positions)
                if positions else [{} for _ in raw]
            )
            for i in range(len(chunk)):
                n = n_options[start + i]
                letter_logits = logits[i, self._letter_ids[:n]]
                logprobs = torch.log_softmax(letter_logits, dim=-1).cpu().numpy()
                extra = {
                    name: torch.stack([h[i, j, :] for h in hs], dim=0).float().cpu().numpy()
                    .astype(np.float16)
                    for name, j in extra_idx[i].items()
                }
                out.append(
                    Readout(
                        hidden=stacked[i].float().cpu().numpy().astype(np.float16),
                        letter_logprobs=logprobs.astype(np.float32),
                        answer_index=int(np.argmax(logprobs)),
                        n_tokens=int(enc["attention_mask"][i].sum()),
                        extra=extra,
                    )
                )
            del res, hs, stacked, logits, enc  # release the batch before the next allocation
        return out

    @staticmethod
    def _extra_positions(raw, offsets, mask, meta, start, positions) -> list[dict[str, int]]:
        """Token index of each requested extra position, per row, in the padded batch."""
        out = []
        for i, text in enumerate(raw):
            offs = offsets[i].tolist()
            valid = [j for j in range(len(offs)) if int(mask[i, j]) == 1]

            def token_at(char, offs=offs, valid=valid):
                for j in valid:
                    a, b = offs[j]
                    if a <= char < b:
                        return j
                return None

            found: dict[str, int] = {}
            if "generation_last" in positions and text:
                j = token_at(len(text) - 1)
                if j is not None:
                    found["generation_last"] = j
            if "stated_answer" in positions:
                gs = (meta[start + i].get("gen_start_char") if meta else None)
                if gs is not None:
                    matches = list(ANSWER_RE.finditer(text, gs))
                    if matches:
                        j = token_at(matches[-1].start(1))
                        if j is not None:
                            found["stated_answer"] = j
            out.append(found)
        return out

    def stress(self, prompt_tokens: int, gen_tokens: int, gen_batch: int, readout_batch: int) -> dict:
        """Worst-case memory: full-length prompts forced to generate the whole budget, then a
        readout over the longest transcript the protocol can produce. For the smoke run only."""
        torch = self.torch
        filler = self.tokenizer.encode("The panel weighed each option in turn. " * 400,
                                       add_special_tokens=False)
        while len(filler) < prompt_tokens + gen_tokens:
            filler = filler * 2
        cuda = self.device.startswith("cuda")
        report = {}
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        ids = torch.tensor([filler[:prompt_tokens]] * gen_batch, device=self.device)
        t = time.time()
        with torch.no_grad():
            g = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    max_new_tokens=gen_tokens, min_new_tokens=gen_tokens,
                                    do_sample=True, temperature=0.7, top_p=0.95,
                                    pad_token_id=self.tokenizer.pad_token_id)
        report["generation"] = {"batch": gen_batch, "prompt_tokens": prompt_tokens,
                                "new_tokens": int(g.shape[1] - prompt_tokens),
                                "seconds": round(time.time() - t, 1), **self._peaks(cuda)}
        del g, ids
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        L = prompt_tokens + gen_tokens
        ids = torch.tensor([filler[:L]] * readout_batch, device=self.device)
        t = time.time()
        with torch.no_grad():
            res = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             output_hidden_states=True, logits_to_keep=1, use_cache=False)
        finite = bool(torch.isfinite(res.hidden_states[-1][:, -1, :]).all())
        report["readout"] = {"batch": readout_batch, "tokens": L, "finite": finite,
                             "seconds": round(time.time() - t, 1), **self._peaks(cuda)}
        del res, ids
        if cuda:
            torch.cuda.empty_cache()
        return report

    def _peaks(self, cuda: bool) -> dict:
        if not cuda:
            return {}
        t = self.torch.cuda
        return {"peak_allocated_gb": round(t.max_memory_allocated() / 1e9, 2),
                "peak_reserved_gb": round(t.max_memory_reserved() / 1e9, 2)}
