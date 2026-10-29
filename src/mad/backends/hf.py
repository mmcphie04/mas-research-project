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
from mad.parsing import parse_answer_field
from mad.permute import LETTERS
from mad.prompts import READOUT_SUFFIX


def _host_memory() -> dict:
    """Host resident memory: current, and the process peak so far (Linux units)."""
    import resource

    out = {"host_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    try:
        with open("/proc/self/statm") as fh:
            out["host_rss_gb"] = round(int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9, 2)
    except OSError:
        pass
    return out


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
        # float16 matmul is unsupported or pathologically slow on most CPUs; compute in fp32 there.
        # Activations are stored as bfloat16 either way (mad.store).
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

        The input is built from token ids, not by re-tokenising the transcript as text:
        ``tokens(prompt) + generated ids + tokens(READOUT_SUFFIX)``. Re-tokenising text can merge
        the response's last token with the suffix (a final newline plus the suffix's newline
        becomes one token), so the model would read a different sequence than it generated.

        ``meta[i]["gen_start_char"]`` is where the agent's own response starts in ``texts[i]``
        (None: the whole text is prompt, as for the peer-exposure state). ``meta[i]
        ["gen_token_ids"]`` are the ids generation produced; without them the response is
        tokenised on its own and ``alignment["replay_exact"]`` is None (unknown).
        Answer positions are searched for ONLY inside the agent's own response.
        """
        torch, mc = self.torch, self.cfg.model
        bs = batch_size or mc.readout_batch_size or mc.batch_size
        positions = [p for p in self.cfg.activations.positions if p != "readout_last"]
        pad_id = self.tokenizer.pad_token_id
        suffix_ids = self.tokenizer.encode(READOUT_SUFFIX, add_special_tokens=False)
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
            seqs, infos = [], []
            for i, text in enumerate(texts[start : start + bs]):
                m = meta[start + i] if meta else {}
                seq, info = self._build_sequence(text, m, suffix_ids, n_options[start + i], positions)
                seqs.append(seq)
                infos.append(info)
            width = max(len(x) for x in seqs)
            ids = torch.tensor([[pad_id] * (width - len(x)) + x for x in seqs])  # left padding
            mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in seqs])
            enc = {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}
            with torch.no_grad():
                # logits_to_keep=1 is not an optimisation, it is a hard requirement. Gemma 3 has
                # a 262,144-token vocabulary, so the default full-sequence logits tensor is
                # batch x seq x 262144 x 4 bytes -- 22.6 GB at batch 24 over a 900-token debate
                # transcript, which OOM-killed an overnight run. Only the last position is ever
                # read, and keeping just that costs 25 MB. use_cache=False: a prefill-only pass
                # has no use for the key/value cache, which would otherwise be allocated.
                res = self.model(**enc, output_hidden_states=True, logits_to_keep=1, use_cache=False)
            # hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, d_model)
            hs = res.hidden_states
            stacked = torch.stack([h[:, -1, :] for h in hs], dim=1)  # (b, L+1, d)
            logits = res.logits[:, -1, :].float()  # seq dim is length 1
            for i, (seq, info) in enumerate(zip(seqs, infos, strict=True)):
                n = n_options[start + i]
                pad = width - len(seq)
                letter_logits = logits[i, self._letter_ids[:n]]
                logprobs = torch.log_softmax(letter_logits, dim=-1).cpu().numpy()
                extra, extra_lp = {}, {}
                for name, j in info["found"].items():
                    extra[name] = torch.stack([h[i, pad + j, :] for h in hs], dim=0).float().cpu().numpy()
                    extra_lp[name] = self.letter_logprobs_from_hidden(hs[-1][i, pad + j, :], n)
                alignment = {
                    "positions": {k: [j, int(seq[j])] for k, j in info["found"].items()},
                    "missing": info["missing"],
                    "replay_exact": info["replay_exact"],
                    "n_prompt_tokens": info["n_prompt"],
                    "n_response_tokens": info["n_gen"],
                    **({"stated_token_id_observed": info["stated_token_id_observed"]}
                       if "stated_token_id_observed" in info else {}),
                    **self._response_logprobs(hs[-1][i], pad, info["n_prompt"], seq),
                }
                out.append(
                    Readout(
                        hidden=stacked[i].float().cpu().numpy(),  # float32; stored as bfloat16
                        letter_logprobs=logprobs.astype(np.float32),
                        answer_index=int(np.argmax(logprobs)),
                        n_tokens=len(seq),
                        extra=extra,
                        extra_logprobs=extra_lp,
                        alignment=alignment,
                    )
                )
            del res, hs, stacked, logits, enc  # release the batch before the next allocation
        return out

    def _response_logprobs(self, h_final, pad: int, n_prompt: int, seq: list[int],
                           chunk: int | None = None) -> dict:
        """Log-probability (temperature 1) the model assigns to each token of its own response,
        summarised as mean / min / sum. Computed from the final hidden states with the output
        projection applied in chunks of positions, never as a full sequence x vocabulary tensor.
        Empty for prompt-only readouts."""
        torch = self.torch
        chunk = chunk or self.cfg.activations.likelihood_chunk
        n_gen = len(seq) - n_prompt - len(self.tokenizer.encode(READOUT_SUFFIX, add_special_tokens=False))
        if n_gen <= 0:
            return {}
        W = self.model.get_output_embeddings().weight
        cfg = getattr(self.model.config, "text_config", self.model.config)
        cap = getattr(cfg, "final_logit_softcapping", None)
        targets = torch.tensor(seq[n_prompt: n_prompt + n_gen], device=W.device)
        out = []
        with torch.no_grad():
            for a in range(0, n_gen, chunk):
                b = min(a + chunk, n_gen)
                h = h_final[pad + n_prompt - 1 + a: pad + n_prompt - 1 + b]  # position t predicts t+1
                z = (h.to(W.dtype) @ W.T).float()
                if cap:
                    z = torch.tanh(z / cap) * cap
                lp = torch.log_softmax(z, dim=-1)
                out.append(lp.gather(1, targets[a:b, None])[:, 0])
                del z, lp
        v = torch.cat(out).cpu().numpy()
        return {"response_logprob_mean": float(v.mean()), "response_logprob_min": float(v.min()),
                "response_logprob_sum": float(v.sum())}

    def _build_sequence(self, text, m, suffix_ids, n_options, positions):
        """Token ids for one readout, and where each requested extra position is in them."""
        tok = self.tokenizer
        gs, gen_ids = m.get("gen_start_char"), m.get("gen_token_ids")
        if gs is None:
            prompt_ids, gen, replay = tok.encode(text, add_special_tokens=False), [], None
        else:
            prompt_ids = tok.encode(text[:gs], add_special_tokens=False)
            if gen_ids is not None:
                gen = list(gen_ids)
                # the stored response omits the token that ended the turn (e.g. <end_of_turn>,
                # a generation stop id that is not in all_special_ids); drop only that one
                while gen and gen[-1] in self._stop_ids | set(tok.all_special_ids):
                    gen.pop()
                replay = tok.decode(gen, skip_special_tokens=True) == text[gs:]
            else:
                gen, replay = tok.encode(text[gs:], add_special_tokens=False), None
        seq = prompt_ids + gen + suffix_ids
        P = len(prompt_ids)
        found, missing, extra_info = {}, {}, {}
        if "generation_last" in positions:
            # last token before the suffix: the last generated token, or for a prompt-only
            # readout (peer exposure) the last prompt token
            found["generation_last"] = P + len(gen) - 1 if (gen or P) else None
            if found["generation_last"] is None:
                del found["generation_last"]
                missing["generation_last"] = "empty_text"
        wants = [p for p in ("stated_answer", "before_stated_answer") if p in positions]
        if wants:
            if gs is None:
                missing.update({p: "no_generated_response" for p in wants})
            else:
                parsed = parse_answer_field(text[gs:], n_options)
                k = self._token_of_char(gen, parsed.letter_start) if parsed.ok else None
                if not parsed.ok:
                    missing.update({p: f"answer_{parsed.status}" for p in wants})
                elif k is None:
                    missing.update({p: "letter_not_aligned" for p in wants})
                elif seq[P + k] != self._letter_ids[parsed.index]:
                    # The token holding the stated letter is not the standalone letter token the
                    # readout scores (e.g. " C" after a space in a permissive "Answer: C"). Its
                    # state and the preceding token's letter distribution would not measure the
                    # same thing, so neither is stored; the observed id is kept for audit.
                    missing.update({p: "letter_token_mismatch" for p in wants})
                    extra_info["stated_token_id_observed"] = int(seq[P + k])
                else:
                    if "stated_answer" in positions:
                        found["stated_answer"] = P + k
                    if "before_stated_answer" in positions:
                        # the token whose output predicts the letter -- inside the agent's own
                        # response, never a peer's or an earlier turn's
                        if k >= 1:
                            found["before_stated_answer"] = P + k - 1
                        else:
                            missing["before_stated_answer"] = "outside_own_response"
        return seq, {"found": found, "missing": missing, "replay_exact": replay,
                     "n_prompt": P, "n_gen": len(gen), **extra_info}

    def _token_of_char(self, gen: list[int], char: int) -> int | None:
        """Index of the generated token that contains character ``char`` of the decoded response,
        found by decoding growing prefixes; None if that token does not contain the letter."""
        dec = self.tokenizer.decode
        prev = ""
        for k in range(len(gen)):
            cur = dec(gen[: k + 1], skip_special_tokens=True)
            if len(cur) > char:
                piece = cur[len(prev):] if cur.startswith(prev) else dec([gen[k]])
                return k if piece.strip() and cur[char] in piece else None
            prev = cur
        return None

    HEADS = ("fp32_head", "bf16_logits")

    def letter_logprobs_ids(self, seqs: list[list[int]], n_options: list[int],
                            batch_size: int | None = None, head: str = "fp32_head",
                            position_ids_from_mask: bool = False) -> list[np.ndarray]:
        """Option-letter log-probabilities at the LAST token of each id sequence (prefill only).

        ``head``:
          fp32_head    the final-layer hidden state at the last position, projected onto the
                       option-letter rows in float32 with Gemma's soft-capping
                       (``letter_logprobs_from_hidden``). The model's own logits tensor is
                       bfloat16, whose spacing at Gemma's soft-capped logit scale (|z| ~ 16-30)
                       is 0.0625-0.125: a one-step rounding difference moves a close two-letter
                       contest by 0.15-0.4 in probability (docs/NEXT_PHASE_CPU_RESULTS.md s11).
          bf16_logits  ``logits_to_keep=1`` logits as the model returns them -- the path of the
                       saved peer_exposure readouts, kept for bridging to them.
        Inputs are token ids built by the caller, never re-tokenised. Left padding, as in
        ``readout``. ``position_ids_from_mask`` passes explicit positions (0 at the first real
        token) instead of the model's default; diagnostics only.
        """
        if head not in self.HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {self.HEADS}")
        torch, mc = self.torch, self.cfg.model
        bs = batch_size or mc.readout_batch_size or mc.batch_size
        pad_id = self.tokenizer.pad_token_id
        out: list[np.ndarray] = []
        for start in range(0, len(seqs), bs):
            chunk = seqs[start: start + bs]
            width = max(len(x) for x in chunk)
            ids = torch.tensor([[pad_id] * (width - len(x)) + list(x) for x in chunk])
            mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in chunk])
            kw = {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device),
                  "logits_to_keep": 1, "use_cache": False}
            if position_ids_from_mask:
                kw["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0).to(self.device)
            with torch.no_grad():
                res = self.model(**kw, output_hidden_states=head == "fp32_head")
            for i in range(len(chunk)):
                n = n_options[start + i]
                if head == "fp32_head":
                    out.append(self.letter_logprobs_from_hidden(res.hidden_states[-1][i, -1, :], n))
                else:
                    lp = torch.log_softmax(res.logits[i, -1, self._letter_ids[:n]].float(), dim=-1)
                    out.append(lp.cpu().numpy().astype(np.float32))
            del res, ids, mask
        return out

    def letter_logprobs_from_hidden(self, h_final, n: int) -> np.ndarray:
        """Option-letter log-probabilities at any position, from that position's FINAL hidden
        state, without computing full-vocabulary logits: only the n letter rows of the output
        projection, plus Gemma's final logit soft-capping when the config sets it.

        ``hidden_states[-1]`` is already after the final norm, so this is the model's own
        output transformation; ``tests/test_backends.py`` checks it reproduces the model's
        logits at the readout position exactly.
        """
        torch = self.torch
        with torch.no_grad():
            W = self.model.get_output_embeddings().weight[self._letter_ids[:n]]
            z = (W.float() @ h_final.float())
            cfg = getattr(self.model.config, "text_config", self.model.config)
            cap = getattr(cfg, "final_logit_softcapping", None)
            if cap:
                z = torch.tanh(z / cap) * cap
            return torch.log_softmax(z, dim=-1).cpu().numpy().astype(np.float32)

    def stress(self, prompt_tokens: int, gen_tokens: int, readout_tokens: int,
               gen_batches: list[int], readout_batches: list[int]) -> dict:
        """Worst-case memory of the REAL workload at each batch size.

        Generation: full-length prompts forced to generate the whole token budget through
        ``model.generate`` (the call ``generate()`` makes). Readout: ``self.readout()`` itself --
        hidden states for every layer, every configured extra position with its option
        probabilities, and the chunked response likelihood -- on synthetic transcripts of the
        longest length the protocol can produce, ending in a valid answer line so the stated-answer
        positions are extracted too. Out-of-memory at a size is recorded, not raised. Peaks:
        GPU allocated and reserved, and host resident memory.
        """
        torch = self.torch
        tok = self.tokenizer
        filler = tok.encode("The panel weighed each option in turn. " * 400, add_special_tokens=False)
        while len(filler) < max(prompt_tokens, readout_tokens):
            filler = filler * 2
        cuda = self.device.startswith("cuda")
        tail = tok.encode("\nAnswer: (B)\nConfidence: 80", add_special_tokens=False)
        resp_ids = filler[: max(gen_tokens - len(tail), 1)] + tail
        suffix = len(tok.encode(READOUT_SUFFIX, add_special_tokens=False))
        prompt_text = tok.decode(filler[: max(readout_tokens - len(resp_ids) - suffix, 1)])
        text = prompt_text + tok.decode(resp_ids)
        meta = {"gen_start_char": len(prompt_text), "gen_token_ids": resp_ids}
        report = {"prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
                  "readout_tokens": readout_tokens, "extraction": {
                      "positions": list(self.cfg.activations.positions),
                      "likelihood_chunk": self.cfg.activations.likelihood_chunk,
                      "path": "HFBackend.readout (hidden states, extra positions, option "
                              "probabilities, response likelihood)"},
                  "generation": [], "readout": []}

        def attempt(fn):
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                extra = fn()
                return {"ok": True, "seconds": round(time.time() - t, 1), **extra,
                        **self._peaks(cuda), **_host_memory()}
            except Exception as exc:  # OutOfMemoryError on CUDA; recorded, not raised
                if "out of memory" not in str(exc).lower() and type(exc).__name__ != "OutOfMemoryError":
                    raise
                return {"ok": False, "error": "out_of_memory", **self._peaks(cuda), **_host_memory()}
            finally:
                if cuda:
                    torch.cuda.empty_cache()

        for b in gen_batches:
            def gen(b=b):
                ids = torch.tensor([filler[:prompt_tokens]] * b, device=self.device)
                with torch.no_grad():
                    g = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            max_new_tokens=gen_tokens, min_new_tokens=gen_tokens,
                                            do_sample=True, temperature=0.7, top_p=0.95,
                                            pad_token_id=tok.pad_token_id)
                return {"new_tokens": int(g.shape[1] - prompt_tokens)}
            report["generation"].append({"batch": b, **attempt(gen)})
        for b in readout_batches:
            def ro(b=b):
                reads = self.readout([text] * b, [10] * b, meta=[meta] * b, batch_size=b)
                r = reads[0]
                finite = all(np.isfinite(x.hidden.astype(np.float32)).all()
                             and all(np.isfinite(v.astype(np.float32)).all() for v in x.extra.values())
                             and np.isfinite(x.letter_logprobs).all() for x in reads)
                return {"finite_all_layers": bool(finite), "readout_tokens_actual": r.n_tokens,
                        "positions_extracted": sorted(r.extra),
                        "positions_missing": r.alignment.get("missing", {}),
                        "likelihood_computed": "response_logprob_mean" in r.alignment,
                        "replay_exact": r.alignment.get("replay_exact")}
            report["readout"].append({"batch": b, **attempt(ro)})
        ok_g = [r["batch"] for r in report["generation"] if r["ok"]]
        ok_r = [r["batch"] for r in report["readout"] if r["ok"] and r.get("finite_all_layers")]
        report["max_ok_generation_batch"] = max(ok_g) if ok_g else None
        report["max_ok_readout_batch"] = max(ok_r) if ok_r else None
        return report

    def _peaks(self, cuda: bool) -> dict:
        if not cuda:
            return {}
        t = self.torch.cuda
        return {"peak_allocated_gb": round(t.max_memory_allocated() / 1e9, 2),
                "peak_reserved_gb": round(t.max_memory_reserved() / 1e9, 2)}
