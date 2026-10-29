"""Tier-0 backend: no model, planted ground truth.

This is the instrument that proves the analysis layer is correct before any GPU time is spent.
It generates a revision distribution with closed-form B/H/U rates and activations carrying a
signal whose strength is a config parameter, so every downstream estimator can be checked
against a known answer.

Two knobs matter most:
  probe_signal    -- instance-level correctness signal. Recoverable within a question.
  difficulty_leak -- question-level signal only. Produces high MARGINAL AUROC and chance
                     CONDITIONAL AUROC, which is exactly the difficulty-detector shortcut of
                     docs/RISKS.md risk 1. Setting this alone validates the diagnostic itself.
"""

from __future__ import annotations

import hashlib

import numpy as np

from mad.backends.base import Backend, Generation, Message, OracleEntry, Readout
from mad.permute import LETTERS


def _seed(*parts) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


class StubBackend(Backend):
    name = "stub"

    def __init__(self, cfg, oracle: dict[str, OracleEntry]):
        self.cfg = cfg
        self.s = cfg.stub
        self.oracle = oracle
        self.n_layers = self.s.n_layers
        self.d_model = self.s.d_model
        self.run_seed = cfg.seed
        rng = _seed("directions", self.d_model, self.run_seed)
        self._u_inst = self._unit(rng.normal(size=self.d_model))
        self._u_diff = self._unit(rng.normal(size=self.d_model))
        depths = np.arange(self.n_layers + 1) / max(self.n_layers, 1)
        self._layer_w = np.exp(-((depths - self.s.signal_peak_layer) ** 2) / (2 * 0.15**2))

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    # ------------------------------------------------------------------ planted ground truth

    def difficulty(self, question_id: str) -> float:
        """Latent per-question difficulty in [-0.5, 0.5]; higher is harder."""
        return float(_seed("difficulty", question_id, self.run_seed).uniform(-0.5, 0.5))

    def p_correct(self, question_id: str) -> float:
        d = self.difficulty(question_id)
        return float(np.clip(self.s.base_accuracy - self.s.accuracy_spread * d, 0.02, 0.98))

    def plan_pre(self, question_id: str, agent_id: int) -> int:
        """Planted round-0 answer, in display index space."""
        entry = self.oracle[question_id]
        rng = _seed("pre", question_id, agent_id, self.run_seed)
        if rng.random() < self.p_correct(question_id):
            return entry.gold_display_index
        wrong = [i for i in range(entry.n_options) if i != entry.gold_display_index]
        return int(rng.choice(wrong))

    def plan_post(
        self, question_id: str, agent_id: int, pre: int, peer_majority: int | None, condition: str
    ) -> int:
        """Planted round-1 answer, in display index space."""
        entry = self.oracle[question_id]
        gold, n = entry.gold_display_index, entry.n_options
        if condition == "independent":  # a fresh first answer, not a revision
            r = _seed("independent", question_id, agent_id, self.run_seed)
            if r.random() < self.p_correct(question_id):
                return gold
            return int(r.choice([i for i in range(n) if i != gold]))
        rng = _seed("post", question_id, agent_id, condition, self.run_seed)
        pre_correct = pre == gold
        rate = self.s.revision_rate_correct if pre_correct else self.s.revision_rate_incorrect
        if condition in ("self_reconsider", "self_structured"):
            rate *= self.s.control_revision_scale
        if rng.random() >= rate:
            return pre

        use_peer = (
            peer_majority is not None and peer_majority != pre and rng.random() < self.s.conformity
        )
        if pre_correct:
            if use_peer:
                return peer_majority
            wrong = [i for i in range(n) if i != pre]
            return int(rng.choice(wrong)) if wrong else pre
        if rng.random() < self.s.switch_to_correct:
            return gold
        others = [i for i in range(n) if i not in (pre, gold)]
        if not others:
            return pre
        if use_peer and peer_majority in others:
            return peer_majority
        return int(rng.choice(others))

    def plan_answer(self, **kw) -> int | None:
        if kw["phase"] == "candidate":  # candidate evaluation: endorsement pulls, gold attracts
            entry = self.oracle[kw["question_id"]]
            r = _seed("candidate", kw["question_id"], kw["item"], self.run_seed).random()
            if kw.get("endorsed", -1) >= 0 and r < 0.35:
                return kw["endorsed"]
            if r < 0.35 + self.p_correct(kw["question_id"]) * 0.65:
                return entry.gold_display_index
            return kw["candidate"]
        if kw["phase"] == "pre":
            return self.plan_pre(kw["question_id"], kw["agent_id"])
        return self.plan_post(
            kw["question_id"],
            kw["agent_id"],
            kw["pre"],
            kw.get("peer_majority"),
            kw.get("condition", "debate"),
        )

    # ------------------------------------------------------------------------- Backend API

    def render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def generate(
        self, prompts: list[str], seed: int, meta: list[dict] | None = None
    ) -> list[Generation]:
        if meta is None:
            raise ValueError("the stub backend requires meta (question_id, agent_id, phase, ...)")
        out = []
        for prompt, m in zip(prompts, meta, strict=True):
            # the prose states the planted answer: the stated answer is the agent's decision
            answer = m["planned_answer"]
            rng = _seed("text", m["question_id"], m["agent_id"], m["phase"], self.run_seed)
            correct = answer == self.oracle[m["question_id"]].gold_display_index
            conf = int(np.clip(rng.normal(72 if correct else 62, 12), 0, 100))
            n_steps = int(rng.integers(2, 5))
            body = "\n".join(
                f"Step {i + 1}: consider the {i + 1}th relevant fact." for i in range(n_steps)
            )
            text = f"{body}\nAnswer: ({LETTERS[answer]})\nConfidence: {conf}"
            out.append(
                Generation(
                    text=text,
                    n_prompt_tokens=max(1, len(prompt) // 4),
                    n_completion_tokens=max(1, len(text) // 4),
                    token_ids=None,
                    stop_reason="stop_token",
                )
            )
        return out

    def readout(
        self, texts: list[str], n_options: list[int], meta: list[dict] | None = None
    ) -> list[Readout]:
        if meta is None:
            raise ValueError("the stub backend requires meta")
        out = []
        for text, n, m in zip(texts, n_options, meta, strict=True):
            qid = m["question_id"]
            entry = self.oracle[qid]
            # the readout is a forward pass, not a parse: take the planted answer directly
            answer = m["planned_answer"]
            if m.get("phase") == "peer_exposure" and "pre_answer" in m:
                # partway between the original and the revision: the agent has read the peers
                # but has not generated its new response yet
                committed = _seed("pe", qid, m["agent_id"], self.run_seed).random()
                if committed >= self.s.peer_exposure_commit:
                    answer = m["pre_answer"]
            correct = answer == entry.gold_display_index
            rng = _seed(
                "acts", qid, m["agent_id"], m["phase"], m.get("condition", "debate"), self.run_seed
            )

            sign = 1.0 if correct else -1.0
            diff_term = -2.0 * self.difficulty(qid)  # high when the question is easy
            hidden = rng.normal(scale=self.s.noise_scale, size=(self.n_layers + 1, self.d_model))
            hidden += np.outer(self._layer_w * self.s.probe_signal * sign, self._u_inst)
            hidden += np.outer(self._layer_w * self.s.difficulty_leak * diff_term, self._u_diff)

            # The readout must COMMIT to the planted answer -- it stands in for a forward pass,
            # not a noisy sample -- so the chosen letter is forced to the top. Only the margin is
            # random, which is what gives readout confidence a realistic (imperfect) correlation
            # with correctness for the baseline to be a fair competitor.
            # occasionally the forced readout commits to a different letter than the prose
            # stated, so the stated/readout disagreement tracking has something to measure
            committed = answer
            if m.get("phase") != "peer_exposure" and n > 1:
                nz = _seed("readout-noise", qid, m["agent_id"], m["phase"],
                           m.get("condition", "debate"), self.run_seed)
                if nz.random() < self.s.freetext_noise:
                    committed = int(nz.choice([i for i in range(n) if i != answer]))
            logits = rng.normal(scale=self.s.confidence_noise, size=n)
            gap = 0.8 + (self.s.confidence_signal if correct else 0.0) + rng.normal(scale=0.6)
            logits[committed] = logits.max() + max(gap, 0.05)
            logprobs = logits - np.log(np.exp(logits).sum())
            out.append(
                Readout(
                    hidden=hidden.astype(np.float32),
                    letter_logprobs=logprobs.astype(np.float32),
                    answer_index=int(np.argmax(logprobs)),
                    n_tokens=max(1, len(text) // 4),
                    **self._extra(text, m, n, hidden, logprobs, rng),
                )
            )
        return out

    def letter_logprobs_ids(self, seqs: list[list[int]], n_options: list[int],
                            batch_size: int | None = None, head: str = "fp32_head",
                            position_ids_from_mask: bool = False) -> list[np.ndarray]:
        """A deterministic function of the id sequence (no model): equal sequences give equal
        distributions, which is all the curve plumbing tests need."""
        out = []
        for seq, n in zip(seqs, n_options, strict=True):
            z = _seed("ids", ",".join(map(str, seq)), self.run_seed).normal(size=n)
            out.append((z - np.log(np.exp(z).sum())).astype(np.float32))
        return out

    def _extra(self, text, m, n, hidden, logprobs, rng) -> dict:
        """Extra positions with the same existence rules as the real backend."""
        from mad.parsing import parse_answer_field

        extra, extra_lp, positions, missing = {}, {}, {}, {}
        for p in self.cfg.activations.positions:
            if p == "readout_last":
                continue
            if p in ("stated_answer", "before_stated_answer"):
                gs = m.get("gen_start_char")
                if gs is None:
                    missing[p] = "no_generated_response"
                    continue
                parsed = parse_answer_field(text[gs:], n)
                if not parsed.ok:
                    missing[p] = f"answer_{parsed.status}"
                    continue
            extra[p] = (hidden + rng.normal(scale=0.1, size=hidden.shape)).astype(np.float32)
            extra_lp[p] = logprobs.astype(np.float32)
            positions[p] = [-1, -1]  # the stub has no tokens
        al = {"positions": positions, "missing": missing, "replay_exact": None}
        if m.get("gen_start_char") is not None:  # a response exists: a stand-in likelihood
            al["response_logprob_mean"] = float(-0.3 - 0.4 * rng.random())
            al["response_logprob_min"] = float(-3.0 * rng.random())
        return {"extra": extra, "extra_logprobs": extra_lp, "alignment": al}
