"""Exploratory probe search on the 12B pilot (CPU only; the pilot pool is excluded from main).
Results and their interpretation: docs/PILOT_ANALYSIS.md section 3.

    python scripts/probe_search.py cache                      # layer-major copy of all states (~7 GB)
    python scripts/probe_search.py screen   --out DIR ...     # layers x positions x transforms x methods
    python scripts/probe_search.py nested   --out DIR --recipes POS:TRANSFORM:METHOD ... --layers ...
    python scripts/probe_search.py nested-ens --out DIR --recipes POS:TRANSFORM:METHOD:A-B ... --fixed R
    python scripts/probe_search.py followup --out DIR --recipe R [--layer L]   # controls for one recipe
    python scripts/probe_search.py combine  --out FILE --runs TAG_seedN ...    # post hoc sum; decisions

Every answer-bearing state is a training example for ONE correctness probe: the 360 round-0
states (shared by all conditions) and the 1,440 revised states of the four conditions -- 1,800
states, always split by question. A single-layer recipe reads one layer at one token position of
the state being scored; an ensemble recipe averages fold-standardized scores over a layer band.

Metrics (out of fold, folds by question; scores from different fold models are compared only in
the per-fold auroc_* averages):
  auroc_all, auroc_pre   per-fold AUROC over all states / round-0 states
  within_panel           pairs of the three agents answering the same prompt, different labels
  within_q               pairs within a question across conditions (a condition-recognising probe
                         could exploit it; reported, not relied on)
  rev_auc                AUROC of score(after) - score(before), beneficial vs harmful, within each
                         condition, averaged -- invariant to constant phase or condition offsets
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONDS = ("debate", "debate_answer_only", "self_reconsider", "misleading_one")
POSITIONS = ("readout_last", "generation_last", "before_stated_answer", "stated_answer")
LP_COLS = {"readout": "readout_logprobs", "generation_last": "pos_generation_last_logprobs",
           "before_stated_answer": "pos_before_stated_answer_logprobs",
           "stated_answer": "pos_stated_answer_logprobs"}


# ----------------------------------------------------------------------------- cache


def build_cache(root: Path, prefix: str, cache: Path, states_only: bool = False) -> None:
    """Layer-major float32 copies of every pre/post/peer_exposure state, plus one table of keys,
    labels and letter-invariant output features. Round-0 states are taken once (from debate)."""
    from mad.analysis.states import StateBank

    cache.mkdir(parents=True, exist_ok=True)
    banks, parts = {}, []
    for c in CONDS:
        b = StateBank(root / f"{prefix}-{c}")
        ix = b.index.copy()
        ix["bank_row"] = np.arange(len(ix))
        if c != "debate":
            ix = ix[ix.phase != "pre"]
        ix["bank"] = c
        banks[c] = b
        parts.append(ix[["bank", "bank_row", "condition", "question_id", "agent_id", "phase", "position"]])
    keys = pd.concat(parts, ignore_index=True)
    keys["question_id"] = keys.question_id.astype(str)
    states = []
    for c in CONDS:
        s = pd.read_parquet(root / f"{prefix}-{c}" / "agent_states.parquet")
        s["question_id"] = s.question_id.astype(str)
        if c != "debate":
            s = s[s.phase != "pre"]
        states.append(s)
    st = pd.concat(states, ignore_index=True)
    st = add_output_features(st)
    keep = ["condition", "question_id", "agent_id", "phase", "answer", "correct", "gold_display",
            "n_options", "subject", "answer_source", "stop_reason"] + [c for c in st.columns if c[:3] in ("of_", "ol_")]
    st[keep].to_parquet(cache / "states.parquet", index=False)
    keys.to_parquet(cache / "keys.parquet", index=False)
    if states_only:
        return
    n_layers = banks["debate"].n_layers
    for L in range(n_layers + 1):
        t = time.time()
        X = np.empty((len(keys), banks["debate"].d_model), np.float32)
        for c in CONDS:
            m = (keys.bank == c).to_numpy()
            X[m] = banks[c].layer(L)[keys.bank_row.to_numpy()[m]]
        if not np.isfinite(X).all():
            raise ValueError(f"non-finite activations at layer {L}")
        np.save(cache / f"L{L:02d}.npy", X)
        print(f"layer {L} {X.shape} {time.time() - t:.1f}s", flush=True)
    (cache / "meta.json").write_text(json.dumps({"n_layers": n_layers, "root": str(root), "prefix": prefix}))


def _lp(series) -> np.ndarray:
    out = np.full((len(series), 10), np.nan)
    for i, v in enumerate(series):
        if isinstance(v, str) and v:
            a = np.array([float(x) for x in v.split(",")])
            out[i, : len(a)] = a
    return out


def add_output_features(st: pd.DataFrame) -> pd.DataFrame:
    """Letter-invariant output features (``of_*``): for each saved option distribution, the
    probability of the state's own answer, the top probability, the top-two margin and the
    entropy; plus response log-likelihood, length and fallback flags."""
    st = st.copy()
    ans = st.answer.to_numpy()
    for name, col in LP_COLS.items():
        if col not in st.columns:
            continue
        lp = _lp(st[col])
        n = st.n_options.to_numpy()
        for i in range(len(lp)):
            lp[i, n[i]:] = np.nan
        ok = np.isfinite(lp).any(1)
        z = np.where(np.isfinite(lp), lp, -np.inf)
        z = z - np.nanmax(np.where(np.isfinite(lp), lp, np.nan), axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(1, keepdims=True)
        srt = -np.sort(-np.nan_to_num(p, nan=0.0), axis=1)
        p_ans = np.array([p[i, a] if ok[i] and 0 <= a < 10 else np.nan for i, a in enumerate(ans)])
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
        st[f"of_{name}_p_answer"] = np.where(ok, p_ans, np.nan)
        st[f"of_{name}_p_max"] = np.where(ok, srt[:, 0], np.nan)
        st[f"of_{name}_margin"] = np.where(ok, srt[:, 0] - srt[:, 1], np.nan)
        st[f"of_{name}_entropy"] = np.where(ok, ent, np.nan)
    for c in ("response_logprob_mean", "response_logprob_min", "n_completion_tokens"):
        st[f"of_{c}"] = st[c].astype(float)
    st["of_capped"] = (st.stop_reason == "length").astype(float)
    st["of_fallback"] = (st.answer_source != "stated").astype(float)
    st["of_verbalized"] = st.verbalized_confidence.astype(float)
    # agreement (output-only, known at decision time; one option order per question, so display
    # letters are comparable across agents): other agents in the same panel with my answer; for
    # revised states, shown peers whose answer equals mine, and whether I changed my answer
    panel = st.question_id.astype(str) + "|" + st.condition + "|" + st.phase
    cnt = st.groupby([panel, st.answer]).answer.transform("size")
    st["of_panel_agree"] = (cnt - 1).astype(float)
    shown = st.shown_peer_answers.fillna("").astype(str)
    st["of_shown_agree"] = [float(sum(int(x) == a for x in v.split(","))) if v else np.nan
                            for v, a in zip(shown, st.answer, strict=True)]
    pre_ans = st[st.phase == "pre"].drop_duplicates(["question_id", "agent_id"]).set_index(
        ["question_id", "agent_id"]).answer
    own_pre = pre_ans.reindex(pd.MultiIndex.from_frame(st[["question_id", "agent_id"]])).to_numpy()
    st["of_revised"] = np.where(st.phase == "post", (st.answer.to_numpy() != own_pre).astype(float), 0.0)
    for name, col in LP_COLS.items():  # letter-specific option log-probabilities ("ol_")
        if col in st.columns:
            lp = _lp(st[col])
            for j in range(10):
                st[f"ol_{name}_{j}"] = lp[:, j]
    return st


# ----------------------------------------------------------------------------- data


class Data:
    """The answer-bearing states (pre and post), their labels and activations at a position."""

    def __init__(self, cache: Path):
        self.cache = cache
        self.keys = pd.read_parquet(cache / "keys.parquet")
        st = pd.read_parquet(cache / "states.parquet")
        st = st[st.phase.isin(["pre", "post"])].reset_index(drop=True)
        st.loc[st.phase == "pre", "condition"] = "round0"
        self.states = st
        self.meta = json.loads((cache / "meta.json").read_text())
        k = self.keys.copy()
        k.loc[k.phase == "pre", "condition"] = "round0"
        self._row = {(c, q, int(a), ph, pos): i for i, (c, q, a, ph, pos) in
                     enumerate(k[["condition", "question_id", "agent_id", "phase", "position"]].itertuples(index=False))}
        self.y = st.correct.to_numpy(int)
        self.q = st.question_id.to_numpy()
        self.groups = pd.factorize(self.q)[0]
        self.panels = panel_ids(st)
        self._layer_cache: dict[int, np.ndarray] = {}
        self._covs: dict[str, np.ndarray] = {}

    def rows(self, position: str) -> np.ndarray:
        return np.array([self._row.get((c, q, int(a), ph, position), -1) for c, q, a, ph in
                         self.states[["condition", "question_id", "agent_id", "phase"]].itertuples(index=False)])

    def layer(self, L: int) -> np.ndarray:
        if L not in self._layer_cache:
            self._layer_cache = {L: np.load(self.cache / f"L{L:02d}.npy", mmap_mode="r")}
        return self._layer_cache[L]

    def X(self, L: int, position: str) -> tuple[np.ndarray, np.ndarray]:
        r = self.rows(position)
        present = r >= 0
        X = np.zeros((len(r), self.layer(L).shape[1]), np.float32)
        X[present] = self.layer(L)[r[present]]
        return X, present

    def covs(self, kind: str) -> dict:
        """Covariates for the residualizing transforms, NaN-filled with missing indicators."""
        out = {}
        for key, sets in (("rtext", ("ot",)), ("rout", ("of", "ol", "ot"))):
            if key in kind:
                if key not in self._covs:
                    M = self.output(sets)
                    miss = ~np.isfinite(M)
                    with np.errstate(all="ignore"):
                        mu = np.nanmean(M, 0)
                    M = np.where(miss, np.where(np.isfinite(mu), mu, 0.0), M)
                    keep = miss.any(0)
                    self._covs[key] = np.hstack([M, miss[:, keep].astype(float)])
                out[key] = self._covs[key]
        return out

    def output(self, sets=("of", "ol", "ot")) -> np.ndarray:
        """Comparator features, by set: ``of`` letter-invariant output summaries and agreement,
        ``ol`` per-letter option log-probabilities, ``ot`` the chosen option's text properties
        (no model involved). All are available at decision time without internals."""
        parts = []
        cols = [c for c in self.states.columns if c[:3] in tuple(f"{x}_" for x in sets if x != "ot")]
        if cols:
            parts.append(self.states[cols].to_numpy(float))
        if "ot" in sets:
            root, prefix = Path(self.meta["root"]), self.meta["prefix"]
            parts.append(option_text_features(self, root / f"{prefix}-questions.jsonl",
                                              root / f"{prefix}-debate" / "agent_states.parquet"))
        return np.hstack(parts)

    def revision_pairs(self) -> pd.DataFrame:
        """(pre_index, post_index, condition, beneficial) for revisions with exactly one correct
        side, over the four conditions."""
        st = self.states.reset_index()
        pre = st[st.phase == "pre"].set_index(["question_id", "agent_id"])
        out = []
        for _, r in st[st.phase == "post"].iterrows():
            p = pre.loc[(r.question_id, r.agent_id)]
            if p.answer != r.answer and (p.correct + r.correct) == 1:
                out.append({"pre": int(p["index"]), "post": int(r["index"]), "condition": r.condition,
                            "beneficial": int(r.correct)})
        return pd.DataFrame(out)


# ----------------------------------------------------------------------------- folds, transforms


def make_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """A random partition of QUESTIONS into k folds (every state of a question in one fold)."""
    u = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(u)
    f = {g: i % k for i, g in enumerate(perm)}
    return np.array([f[g] for g in groups])


def panel_ids(st: pd.DataFrame) -> np.ndarray:
    """One id per panel: (question, condition, phase) -- the three agents answering together."""
    return pd.factorize(st.question_id + "|" + st.condition + "|" + st.phase)[0]


def transform(X: np.ndarray, kind: str, tr: np.ndarray, letters: np.ndarray, panels: np.ndarray,
              covs: dict | None = None) -> np.ndarray:
    """``letter``: subtract the TRAINING mean of states whose answer has the same displayed letter
    (removes "which letter am I about to say", which a linear probe cannot turn into confidence).
    ``panel``: subtract the mean of the state's own panel (label-free; leaves what distinguishes
    an agent from its panel). Both may be combined (``letter+panel``).
    ``rtext`` / ``rout``: regress the activations on the option-text features / on ALL output-only
    features (text, summaries, agreement, per-letter log-probabilities), fitted on training rows,
    and keep the residual -- a linear probe on it cannot express any linear function of them."""
    X = X.astype(np.float32, copy=True)
    for key in ("rtext", "rout"):
        if key in kind:
            Z = covs[key]
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
            Zt = np.hstack([np.ones((len(Z), 1)), (Z - mu) / sd])
            B, *_ = np.linalg.lstsq(Zt[tr], X[tr], rcond=None)
            X -= (Zt @ B).astype(np.float32)
    if "letter" in kind:
        g = X[tr].mean(0)
        for a in np.unique(letters):
            m = letters == a
            mt = m & tr
            X[m] -= X[mt].mean(0) if mt.sum() >= 5 else g
    if "panel" in kind:
        df = pd.DataFrame(X)
        X = (df - df.groupby(panels).transform("mean")).to_numpy(np.float32)
    return X


# ----------------------------------------------------------------------------- methods


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_score(method: str, Xtr, ytr, Xte) -> np.ndarray:
    """Scores for the test rows (higher = more likely correct). Everything fitted on train."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    Xtr, Xte = _std(Xtr, Xte)
    if method == "meandiff":
        w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        return Xte @ w
    if method == "lda":
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
        return m.decision_function(Xte)
    if method.startswith("pca"):
        rest = method[3:]
        k, C = (int(rest.split("c")[0]), float(rest.split("c")[1])) if "c" in rest else (int(rest), 1.0)
        p = PCA(n_components=min(k, len(Xtr) - 1), svd_solver="randomized", random_state=0).fit(Xtr)
        Ztr, Zte = _std(p.transform(Xtr), p.transform(Xte))
        return LogisticRegression(C=C, max_iter=4000).fit(Ztr, ytr).decision_function(Zte)
    if method.startswith("lr"):
        return LogisticRegression(C=float(method[2:]), max_iter=4000).fit(Xtr, ytr).decision_function(Xte)
    raise ValueError(method)


def output_score(method: str, Otr, ytr, Ote) -> np.ndarray:
    """Output-only comparator on the letter-invariant output features."""
    if method == "out_hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                              random_state=0).fit(Otr, ytr).predict_proba(Ote)[:, 1]
    from sklearn.linear_model import LogisticRegression
    mu = np.nanmean(Otr, 0)
    fill = lambda M: np.hstack([np.where(np.isfinite(M), M, mu), (~np.isfinite(M)).astype(float)])  # noqa: E731
    Atr, Ate = _std(fill(Otr), fill(Ote))
    return LogisticRegression(C=1.0, max_iter=4000).fit(Atr, ytr).decision_function(Ate)


# ----------------------------------------------------------------------------- metrics


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def pair_auroc(y, s, groups) -> tuple[float, int]:
    """AUROC over pairs with different labels inside the same group (ties count half)."""
    num, den = 0.0, 0
    for g in np.unique(groups):
        m = groups == g
        yy, ss = y[m], s[m]
        pos, neg = ss[yy == 1], ss[yy == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return (num / den if den else float("nan")), den


def metrics(D: Data, s: np.ndarray, folds: np.ndarray, present: np.ndarray, pairs: pd.DataFrame) -> dict:
    """Ranking metrics that never compare scores from different fold models except auroc_*
    (averaged per fold). ``within_q`` mixes conditions and phases of a question (a probe that
    recognises the condition from the prompt could exploit it); ``within_panel`` compares only
    the three agents answering the same prompt. ``rev_auc``: AUROC of score(after) - score(before)
    for beneficial vs harmful revisions, within each condition, averaged over conditions --
    invariant to any constant offset between phases or conditions."""
    y, st = D.y, D.states
    pre = (st.phase == "pre").to_numpy()
    out = {}
    fl = np.unique(folds[present])
    out["auroc_all"] = float(np.nanmean([auroc(y[(folds == f) & present], s[(folds == f) & present]) for f in fl]))
    out["auroc_pre"] = float(np.nanmean([auroc(y[(folds == f) & present & pre], s[(folds == f) & present & pre])
                                         for f in fl]))
    out["within_q"], out["within_q_pairs"] = pair_auroc(y[present], s[present], D.groups[present])
    panels = D.panels
    out["within_panel"], out["within_panel_pairs"] = pair_auroc(y[present], s[present], panels[present])
    m = present & pre
    out["within_panel_pre"], out["within_panel_pre_pairs"] = pair_auroc(y[m], s[m], panels[m])
    ok = present[pairs.pre.to_numpy()] & present[pairs.post.to_numpy()]
    p = pairs[ok]
    d = s[p.post.to_numpy()] - s[p.pre.to_numpy()]
    per = {c: auroc(p.beneficial.to_numpy()[(p.condition == c).to_numpy()], d[(p.condition == c).to_numpy()])
           for c in sorted(p.condition.unique())}
    out["rev_auc"] = float(np.nanmean(list(per.values())))
    out["rev_auc_debate"] = per.get("debate", float("nan"))
    out["rev_auc_pooled"] = auroc(p.beneficial.to_numpy(), d)
    out["rev_n"] = int(len(p))
    return out


def run_config(D, X, present, letters, panels, kind, method, folds) -> np.ndarray:
    """Out-of-fold scores for one (transform, method); NaN where the state is absent."""
    s = np.full(len(D.y), np.nan)
    for f in np.unique(folds[folds >= 0]):  # negative = not part of this cross-validation
        te = (folds == f) & present
        tr = (folds != f) & present
        Xt = transform(X, kind, tr, letters, panels, D.covs(kind))
        s[te] = fit_score(method, Xt[tr], D.y[tr], Xt[te])
    return s


def screen_task(cache, L, position, kinds, methods, seeds, k):
    import sklearn.decomposition  # noqa: F401  (load every BLAS before limiting threads)
    import sklearn.discriminant_analysis  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        D = Data(Path(cache))
        X, present = D.X(L, position)
        letters = D.states.answer.to_numpy()
        panels = D.panels
        pairs = D.revision_pairs()
        rows, scores = [], {}
        for seed in seeds:
            folds = make_folds(D.groups, k, seed)
            for kind in kinds:
                for method in methods:
                    t = time.time()
                    s = run_config(D, X, present, letters, panels, kind, method, folds)
                    scores[f"{L}|{position}|{kind}|{method}|{seed}"] = s.astype(np.float32)
                    rows.append({"layer": L, "position": position, "transform": kind, "method": method,
                                 "seed": seed, "n": int(present.sum()), "secs": round(time.time() - t, 1),
                                 **metrics(D, s, folds, present, pairs)})
        return rows, scores


def output_rows(cache, seeds, k):
    D = Data(Path(cache))
    Out = D.output()
    present = np.ones(len(D.y), bool)
    pairs = D.revision_pairs()
    rows, scores = [], {}
    for seed in seeds:
        folds = make_folds(D.groups, k, seed)
        for method in ("out_lr", "out_hgb"):
            s = np.full(len(D.y), np.nan)
            for f in np.unique(folds):
                te, tr = folds == f, folds != f
                s[te] = output_score(method, Out[tr], D.y[tr], Out[te])
            scores[f"-1|output|-|{method}|{seed}"] = s.astype(np.float32)
            rows.append({"layer": -1, "position": "output", "transform": "-", "method": method, "seed": seed,
                         "n": int(len(D.y)), **metrics(D, s, folds, present, pairs)})
    return rows, scores


# ----------------------------------------------------------------------------- layer ensembles


def parse_ens(recipe: str):
    """``position:transform:method:a-b[:step]`` -> (position, transform, method, layers)."""
    parts = recipe.split(":")
    a, b = (int(x) for x in parts[3].split("-"))
    step = int(parts[4]) if len(parts) > 4 else 3
    return parts[0], parts[1], parts[2], list(range(a, b + 1, step))


def ensemble_config(D, recipe, present_rows, folds) -> np.ndarray:
    """Out-of-fold scores averaged over a band of layers: each layer's held-out scores are
    standardized within the fold (label-free) and averaged, so no single layer is chosen."""
    position, kind, method, layers = parse_ens(recipe)
    acc, n = np.zeros(len(D.y)), 0
    for L in layers:
        X, present = D.X(L, position)
        s = run_config(D, X, present & present_rows, D.states.answer.to_numpy(), D.panels, kind, method, folds)
        for f in np.unique(folds[folds >= 0]):
            m = (folds == f) & np.isfinite(s)
            if m.sum() > 1:
                s[m] = (s[m] - s[m].mean()) / (s[m].std() + 1e-9)
        acc = acc + s
        n += 1
    return acc / n


def ens_scores(D, recipe, folds, train_exclude=None) -> np.ndarray:
    """Layer-ensemble scores for every fold, optionally never training on ``train_exclude``
    rows (still scoring them): used for leave-one-condition-out."""
    position, kind, method, layers = parse_ens(recipe)
    excl = np.zeros(len(D.y), bool) if train_exclude is None else train_exclude
    acc = np.zeros(len(D.y))
    for L in layers:
        X, present = D.X(L, position)
        s = np.full(len(D.y), np.nan)
        for f in np.unique(folds[folds >= 0]):
            tr = (folds != f) & (folds >= 0) & present & ~excl
            te = (folds == f) & present
            Xt = transform(X, kind, tr, D.states.answer.to_numpy(), D.panels, D.covs(kind))
            z = fit_score(method, Xt[tr], D.y[tr], Xt[te])
            s[te] = (z - z.mean()) / (z.std() + 1e-9)
        acc = acc + s
    return acc / len(layers)


def _ens_inner(cache, recipe, outer, seed, k_inner):
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        D = Data(Path(cache))
        pairs, rows = D.revision_pairs(), []
        for f in np.unique(outer):
            tr = outer != f
            inner = np.where(tr, make_folds(D.groups, k_inner, seed + 1000 + int(f)), -1)
            sc = ensemble_config(D, recipe, tr, inner)
            rows.append({"fold": int(f), "recipe": recipe, **metrics(D, sc, inner, tr & np.isfinite(sc), pairs)})
        return rows


def _ens_outer(cache, recipe, outer):
    """Outer scores of one ensemble recipe for every outer fold (fold f's scores come from a
    model trained without fold f)."""
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        D = Data(Path(cache))
        return ensemble_config(D, recipe, np.ones(len(D.y), bool), outer)


def nested_ens(cache, recipes, criteria, seed, k_outer=5, k_inner=5, jobs=10):
    from joblib import Parallel, delayed
    D = Data(Path(cache))
    outer = make_folds(D.groups, k_outer, seed)
    res = Parallel(n_jobs=jobs)(delayed(_ens_inner)(cache, r, outer, seed, k_inner) for r in recipes)
    inner_tab = pd.DataFrame([row for rr in res for row in rr])
    outer_scores = dict(zip(recipes, Parallel(n_jobs=jobs)(delayed(_ens_outer)(cache, r, outer) for r in recipes),
                            strict=True))
    out = {}
    for crit in criteria:
        chosen = inner_tab.loc[inner_tab.groupby("fold")[crit].idxmax()]
        s = np.full(len(D.y), np.nan)
        for row in chosen.itertuples():
            m = outer == row.fold
            s[m] = outer_scores[row.recipe][m]
        out[crit] = (s, chosen)
    return out, outer, inner_tab, outer_scores


# ----------------------------------------------------------------------------- nested selection


def _inner_task(cache, L, position, recipes, outer, seed, k_inner):
    """Inner-CV metrics of every recipe at one (layer, position), for every outer fold."""
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        D = Data(Path(cache))
        X, present = D.X(L, position)
        letters, pairs = D.states.answer.to_numpy(), D.revision_pairs()
        rows = []
        for f in np.unique(outer):
            tr = outer != f
            inner = np.where(tr, make_folds(D.groups, k_inner, seed + 1000 + int(f)), -1)
            for r in recipes:
                _, kind, method = r.split(":")
                sc = run_config(D, X, present & tr, letters, D.panels, kind, method, inner)
                rows.append({"fold": int(f), "recipe": r, "layer": L, **metrics(D, sc, inner, present & tr, pairs)})
        return rows


def _outer_fit(cache, recipe, L, tr, te):
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        D = Data(Path(cache))
        position, kind, method = recipe.split(":")
        X, present = D.X(L, position)
        trm, tem = tr & present, te & present
        Xt = transform(X, kind, trm, D.states.answer.to_numpy(), D.panels, D.covs(kind))
        s = np.full(len(D.y), np.nan)
        s[tem] = fit_score(method, Xt[trm], D.y[trm], Xt[tem])
        return s


def nested(cache, recipes, layers, criteria, seed, k_outer=5, k_inner=5, jobs=10):
    """Outer folds by question; inside each outer training set, every (recipe, layer) is scored by
    inner grouped CV, and for each criterion the best candidate is refitted on the whole outer
    training set and applied to the outer test questions, which influence nothing."""
    from joblib import Parallel, delayed
    D = Data(Path(cache))
    outer = make_folds(D.groups, k_outer, seed)
    by_pos = {}
    for r in recipes:
        by_pos.setdefault(r.split(":")[0], []).append(r)
    res = Parallel(n_jobs=jobs, verbose=2)(delayed(_inner_task)(cache, L, pos, rs, outer, seed, k_inner)
                                           for pos, rs in by_pos.items() for L in layers)
    inner_tab = pd.DataFrame([row for rr in res for row in rr])
    out = {}
    for crit in criteria:
        chosen = inner_tab.loc[inner_tab.groupby("fold")[crit].idxmax()]
        fits = Parallel(n_jobs=jobs)(delayed(_outer_fit)(cache, row.recipe, int(row.layer), outer != row.fold,
                                                         outer == row.fold) for row in chosen.itertuples())
        s = np.full(len(D.y), np.nan)
        for row, sf in zip(chosen.itertuples(), fits, strict=True):
            m = outer == row.fold
            s[m] = sf[m]
        out[crit] = (s, chosen)
    return out, outer, inner_tab


def _boot_values(D, s, folds, present, pairs, W) -> dict:
    """Metric values for score vector ``s`` under each row of question weights ``W``."""
    from sklearn.metrics import roc_auc_score
    y, g = D.y, D.groups
    nq = g.max() + 1
    pre = (D.states.phase == "pre").to_numpy()

    def per_q(mask):
        num, den = np.zeros(nq), np.zeros(nq)
        for pnl in np.unique(D.panels[mask]):
            m = mask & (D.panels == pnl)
            pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
            if len(pos) and len(neg):
                d = pos[:, None] - neg[None, :]
                num[g[m][0]] += (d > 0).sum() + 0.5 * (d == 0).sum()
                den[g[m][0]] += d.size
        return num, den

    wp, wpp = per_q(present), per_q(present & pre)
    ok = present[pairs.pre.to_numpy()] & present[pairs.post.to_numpy()]
    p = pairs[ok]
    d = s[p.post.to_numpy()] - s[p.pre.to_numpy()]
    pq, ben = g[p.pre.to_numpy()], p.beneficial.to_numpy()
    conds = sorted(p.condition.unique())

    def wauc(yy, ss, ww):
        return roc_auc_score(yy, ss, sample_weight=ww) if 0 < (ww * yy).sum() < ww.sum() else np.nan

    out = {k: [] for k in ("auroc_all", "auroc_pre", "within_panel", "within_panel_pre", "rev_auc", "rev_auc_debate")}
    for w in W:
        aa, ap = [], []
        for f in np.unique(folds[present]):
            m = (folds == f) & present
            aa.append(wauc(y[m], s[m], w[g[m]]))
            m2 = m & pre
            ap.append(wauc(y[m2], s[m2], w[g[m2]]))
        out["auroc_all"].append(np.nanmean(aa))
        out["auroc_pre"].append(np.nanmean(ap))
        out["within_panel"].append((w @ wp[0]) / (w @ wp[1]))
        out["within_panel_pre"].append((w @ wpp[0]) / (w @ wpp[1]))
        per = [wauc(ben[(p.condition == c).to_numpy()], d[(p.condition == c).to_numpy()],
                    w[pq[(p.condition == c).to_numpy()]]) for c in conds]
        out["rev_auc"].append(np.nanmean(per))
        out["rev_auc_debate"].append(per[conds.index("debate")] if "debate" in conds else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def compare(D, scores: dict, folds, pairs, refs=(), n_boot=2000, seed=0) -> dict:
    """Point values and question-bootstrap intervals for every scorer, on the states every scorer
    covers, plus paired intervals of (scorer - ref) from the SAME resampled questions."""
    present = np.logical_and.reduce([np.isfinite(v) for v in scores.values()])
    nq = D.groups.max() + 1
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones(nq), *[np.bincount(rng.integers(0, nq, nq), minlength=nq) for _ in range(n_boot)]])
    vals = {k: _boot_values(D, v, folds, present, pairs, W) for k, v in scores.items()}
    q = lambda a: [float(np.nanquantile(a[1:], 0.025)), float(np.nanquantile(a[1:], 0.975))]  # noqa: E731
    out = {"n_states": int(present.sum()), "scorers": {}}
    for k, v in vals.items():
        out["scorers"][k] = {m: {"value": float(a[0]), "ci95": q(a)} for m, a in v.items()}
        for ref in refs:
            if k != ref:
                out["scorers"][k]["minus_" + ref] = {m: {"value": float(a[0] - vals[ref][m][0]),
                                                          "ci95": q(a - vals[ref][m])} for m, a in v.items()}
    return out


def option_text_features(D, questions_path, states_path) -> np.ndarray:
    """Properties of the CHOSEN option's text only (no model state): its length, its length
    relative to the question's options, whether it is the longest, its original and displayed
    position. A probe that beat outputs only through these would be reading answer content."""
    qs = {}
    for line in Path(questions_path).read_text().splitlines():
        r = json.loads(line)
        qs[str(r["question_id"])] = r["options"]
    ref = pd.read_parquet(states_path, columns=["question_id", "permutation"])
    perm = {str(q): [int(x) for x in p.split(",")] for q, p in zip(ref.question_id, ref.permutation, strict=True)}
    rows = []
    for q, a in zip(D.states.question_id, D.states.answer, strict=True):
        opts, order = qs[q], perm[q]
        lens = np.array([len(o) for o in opts], float)
        orig = order[int(a)] if 0 <= int(a) < len(order) else 0
        L = lens[orig]
        rows.append([L, (L - lens.mean()) / (lens.std() + 1e-6), float(L == lens.max()),
                     float((lens > L).sum()), float(orig), float(a), float(len(opts))])
    return np.array(rows)


def _train_pairs(D, rows, rev_pairs):
    """(correct, incorrect) index pairs among ``rows``: mixed-label pairs within a panel, and
    revisions with exactly one correct side (same agent before and after)."""
    good, bad = [], []
    for pnl in np.unique(D.panels[rows]):
        idx = np.flatnonzero(rows & (D.panels == pnl))
        pos, neg = idx[D.y[idx] == 1], idx[D.y[idx] == 0]
        for i in pos:
            for j in neg:
                good.append(i)
                bad.append(j)
    for p in rev_pairs.itertuples():
        if rows[p.pre] and rows[p.post]:
            i, j = (p.post, p.pre) if p.beneficial else (p.pre, p.post)
            good.append(i)
            bad.append(j)
    return np.array(good, int), np.array(bad, int)


def _fit_stack(D, Z, tr, te, mode, rev_pairs) -> np.ndarray:
    """``abs``: logistic regression of correctness on Z. ``pair``: logistic regression on the
    DIFFERENCE of Z between the correct and the incorrect member of each training pair (no
    intercept) -- the weights that rank states inside a panel or across a revision, which is the
    decision a gate makes."""
    from sklearn.linear_model import LogisticRegression
    s = np.full(len(D.y), np.nan)
    ok = np.isfinite(Z).all(1)
    trm, tem = tr & ok, te & ok
    m_, sd_ = Z[trm].mean(0), Z[trm].std(0) + 1e-6
    Zs = (Z - m_) / sd_
    if mode == "abs":
        s[tem] = LogisticRegression(C=1.0, max_iter=4000).fit(Zs[trm], D.y[trm]).decision_function(Zs[tem])
        return s
    good, bad = _train_pairs(D, trm, rev_pairs)
    dZ = Zs[good] - Zs[bad]
    w = LogisticRegression(C=1.0, fit_intercept=False, max_iter=4000).fit(
        np.vstack([dZ, -dZ]), np.r_[np.ones(len(dZ)), np.zeros(len(dZ))]).coef_[0]
    s[tem] = Zs[tem] @ w
    return s


def _fill(M, mu):
    return np.hstack([np.where(np.isfinite(M), M, mu), (~np.isfinite(M)).astype(float)])


def stacked_ens_oof(D, chosen, outer, act_outer, seed, k_inner=5, sets=("of", "ot"), mode="abs") -> np.ndarray:
    """stacked_oof for layer-ensemble recipes (training-row activation scores out of fold)."""
    feats, rev = D.output(sets), D.revision_pairs()
    s = np.full(len(D.y), np.nan)
    for row in chosen.itertuples():
        f = row.fold
        tr, te = outer != f, outer == f
        inner = np.where(tr, make_folds(D.groups, k_inner, seed + 1000 + int(f)), -1)
        a = np.where(tr, ensemble_config(D, row.recipe, tr, inner), act_outer)
        Z = np.hstack([a[:, None], _fill(feats, np.nanmean(feats[tr], 0))])
        s[te] = _fit_stack(D, Z, tr, te, mode, rev)[te]
    return s


def output_pair_oof(D, folds, sets=("of", "ot")) -> np.ndarray:
    """Output-only pairwise model: the like-for-like comparator for the pairwise stack."""
    M, rev = D.output(sets), D.revision_pairs()
    s = np.full(len(D.y), np.nan)
    for f in np.unique(folds[folds >= 0]):
        te, tr = folds == f, (folds != f) & (folds >= 0)
        s[te] = _fit_stack(D, _fill(M, np.nanmean(M[tr], 0)), tr, te, "pair", rev)[te]
    return s


def output_oof(D, folds, method, sets=("of", "ol", "ot")) -> np.ndarray:
    out_feats = D.output(sets)
    s = np.full(len(D.y), np.nan)
    for f in np.unique(folds[folds >= 0]):
        te, tr = folds == f, (folds != f) & (folds >= 0)
        s[te] = output_score(method, out_feats[tr], D.y[tr], out_feats[te])
    return s


def stacked_oof(cache, D, chosen, outer, act_outer, seed, k_inner=5, sets=("of", "ot"), mode="abs") -> np.ndarray:
    """Activation score + output features in one model (``abs`` or ``pair``, see _fit_stack).
    For each outer fold the activation score of TRAINING rows is itself out of fold (inner CV
    with the fold's chosen candidate), so the stacker never sees in-sample activation scores."""
    feats, rev = D.output(sets), D.revision_pairs()
    s = np.full(len(D.y), np.nan)
    for row in chosen.itertuples():
        f = row.fold
        tr, te = outer != f, outer == f
        position, kind, method = row.recipe.split(":")
        X, present = D.X(int(row.layer), position)
        inner = np.where(tr, make_folds(D.groups, k_inner, seed + 1000 + int(f)), -1)
        a_tr = run_config(D, X, present & tr, D.states.answer.to_numpy(), D.panels, kind, method, inner)
        a = np.where(tr, a_tr, act_outer)
        Z = np.hstack([a[:, None], _fill(feats, np.nanmean(feats[tr], 0))])
        s[te] = _fit_stack(D, Z, tr, te, mode, rev)[te]
    return s


def decisions(D, s, condition="debate") -> pd.DataFrame:
    """Per question, correctness of panel answers under rules that use scores ``s`` with no
    tuned threshold (ties by the preregistered vote tie-break):
      original_vote   plurality of the three first answers (no scores)
      final_vote      plurality of the three revised answers (no scores)
      gate            each agent keeps its revision only if score(revised) > score(first)
      pick_first      the first answer of the highest-scoring agent
      best_of_six     the highest-scoring of all six answers"""
    from mad.analysis.labels import panel_vote
    st = D.states.reset_index(drop=True)
    pre = st[st.phase == "pre"]
    post = st[(st.phase == "post") & (st.condition == condition)]
    rows = []
    for q, g0 in pre.groupby("question_id"):
        g0 = g0.sort_values("agent_id")
        g1 = post[post.question_id == q].sort_values("agent_id")
        gold = int(g0.gold_display.iloc[0])
        a0, a1 = g0.answer.tolist(), g1.answer.tolist()
        s0, s1 = s[g0.index.to_numpy()], s[g1.index.to_numpy()]
        conf = [0.0] * 3
        gated = [b if (np.isfinite(y) and np.isfinite(x) and y > x) else a
                 for a, b, x, y in zip(a0, a1, s0, s1, strict=True)]
        allans, alls = a0 + a1, np.r_[s0, s1]
        vote0 = panel_vote(a0, conf, None)  # used when a panel has no scores at all
        rows.append({"question_id": q,
                     "original_vote": int(panel_vote(a0, conf, None) == gold),
                     "final_vote": int(panel_vote(a1, conf, a0) == gold),
                     "gate": int(panel_vote(gated, conf, a0) == gold),
                     "pick_first": int((a0[int(np.nanargmax(s0))] if np.isfinite(s0).any() else vote0) == gold),
                     "best_of_six": int((allans[int(np.nanargmax(alls))] if np.isfinite(alls).any() else vote0) == gold)})
    return pd.DataFrame(rows).set_index("question_id")


def loco_masked(D, X, present, kind, method, folds, post_c) -> np.ndarray:
    """Leave one condition out: rows in ``post_c`` (a held-out condition's revised states) are
    never trained on (nor is any state of the test questions), but are scored."""
    s = np.full(len(D.y), np.nan)
    for f in np.unique(folds):
        tr = (folds != f) & present & ~post_c
        te = (folds == f) & present
        Xt = transform(X, kind, tr, D.states.answer.to_numpy(), D.panels, D.covs(kind))
        s[te] = fit_score(method, Xt[tr], D.y[tr], Xt[te])
    return s


def matched_pairs(D, s, present, questions_path, states_path, tol=0.5) -> tuple[float, int]:
    """Within-panel AUROC over pairs whose two chosen options are both (or both not) the longest
    and differ in relative length by less than ``tol`` SD -- answer length cannot separate them."""
    T = option_text_features(D, questions_path, states_path)
    y, num, den = D.y, 0.0, 0
    for pnl in np.unique(D.panels[present]):
        idx = np.flatnonzero(present & (D.panels == pnl))
        for i in idx:
            for j in idx:
                if y[i] == 1 and y[j] == 0 and T[i, 2] == T[j, 2] and abs(T[i, 1] - T[j, 1]) < tol:
                    num += (s[i] > s[j]) + 0.5 * (s[i] == s[j])
                    den += 1
    return (num / den if den else float("nan")), den


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--root", default="runs/colab")
    c.add_argument("--prefix", default="pilot12b")
    c.add_argument("--cache", default="runs/cache/probe12b")
    c.add_argument("--states-only", action="store_true", help="rebuild the state table, not the layers")
    s = sub.add_parser("screen")
    s.add_argument("--cache", default="runs/cache/probe12b")
    s.add_argument("--out", required=True)
    s.add_argument("--layers", type=int, nargs="+", default=list(range(0, 49, 4)))
    s.add_argument("--positions", nargs="+", default=list(POSITIONS))
    s.add_argument("--transforms", nargs="+", default=["raw", "letter", "panel", "letter+panel"])
    s.add_argument("--methods", nargs="+", default=["meandiff", "lda", "pca16", "pca64", "pca256c0.01"])
    s.add_argument("--seeds", type=int, nargs="+", default=[0])
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--jobs", type=int, default=10)
    s.add_argument("--tag", default="screen")
    n = sub.add_parser("nested")
    n.add_argument("--cache", default="runs/cache/probe12b")
    n.add_argument("--out", required=True)
    n.add_argument("--recipes", nargs="+", required=True, help="position:transform:method")
    n.add_argument("--layers", type=int, nargs="+", required=True)
    n.add_argument("--criteria", nargs="+", default=["auroc_all", "within_panel", "rev_auc"])
    n.add_argument("--seeds", type=int, nargs="+", default=[0])
    n.add_argument("--n-boot", type=int, default=2000)
    n.add_argument("--jobs", type=int, default=11)
    n.add_argument("--tag", default="nested")
    fu = sub.add_parser("followup")
    fu.add_argument("--cache", default="runs/cache/probe12b")
    fu.add_argument("--out", required=True)
    fu.add_argument("--recipe", required=True, help="position:transform:method")
    fu.add_argument("--layer", type=int, default=None, help="omit for a layer-ensemble recipe")
    fu.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    fu.add_argument("--n-boot", type=int, default=2000)
    fu.add_argument("--tag", default="followup")
    ne = sub.add_parser("nested-ens")
    ne.add_argument("--cache", default="runs/cache/probe12b")
    ne.add_argument("--out", required=True)
    ne.add_argument("--recipes", nargs="+", required=True, help="position:transform:method:a-b[:step]")
    ne.add_argument("--fixed", default=None, help="one recipe declared in advance: reported without selection")
    ne.add_argument("--criteria", nargs="+", default=["auroc_all", "within_panel", "rev_auc"])
    ne.add_argument("--seeds", type=int, nargs="+", default=[0])
    ne.add_argument("--n-boot", type=int, default=2000)
    ne.add_argument("--jobs", type=int, default=11)
    ne.add_argument("--tag", default="nested_ens")
    cb = sub.add_parser("combine", help="post hoc: equal-weight sum of output and activation scores; panel decisions")
    cb.add_argument("--cache", default="runs/cache/probe12b")
    cb.add_argument("--out", required=True)
    cb.add_argument("--runs", nargs="+", required=True, help="score files: TAG_seedN (folds key 'folds' or 'outer')")
    cb.add_argument("--act", default="act[fixed]", help="activation score key (e.g. act or act[fixed])")
    cb.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.cmd == "combine":
        D = Data(Path(args.cache))
        pairs = D.revision_pairs()

        def zf(v, folds):
            v = v.copy()
            for f in np.unique(folds):
                m = (folds == f) & np.isfinite(v)
                v[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-9)
            return v

        report, rng = {}, np.random.default_rng(0)
        for run in args.runs:
            z = np.load(Path(args.cache) / f"scores_{run}.npz")
            folds = z["folds"] if "folds" in z.files else z["outer"]
            act_key = args.act if args.act in z.files else "act"
            sc = {k: z[k] for k in ("out+text_lr", "out+text_hgb")}
            sc["act"] = z[act_key]
            sc["comb_lr"] = zf(sc["out+text_lr"], folds) + zf(sc["act"], folds)
            sc["comb_hgb"] = zf(sc["out+text_hgb"], folds) + zf(sc["act"], folds)
            seed = int(run.rsplit("seed", 1)[1])
            rep = {"comparison": compare(D, sc, folds, pairs, refs=("out+text_lr", "out+text_hgb"), n_boot=args.n_boot,
                                         seed=seed), "decisions": {}}
            dec = {k: decisions(D, v) for k, v in sc.items()}
            for k, d in dec.items():
                rep["decisions"][k] = {}
                for rule in ("gate", "pick_first", "best_of_six"):
                    diff = (d[rule] - d["original_vote"]).to_numpy(float)
                    boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1)
                    rep["decisions"][k][rule] = {"correct": int(d[rule].sum()), "original_vote": int(d["original_vote"].sum()),
                                                 "minus_original": float(diff.mean()),
                                                 "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}
            report[run] = rep
            print(run, {k: {r: v["correct"] for r, v in d.items()} for k, d in rep["decisions"].items()}, flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        return 0
    if args.cmd == "nested-ens":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        D = Data(Path(args.cache))
        pairs = D.revision_pairs()
        recipes = list(dict.fromkeys(args.recipes + ([args.fixed] if args.fixed else [])))
        for seed in args.seeds:
            t = time.time()
            res, outer, inner_tab, outer_scores = nested_ens(args.cache, recipes, args.criteria, seed, jobs=args.jobs)
            inner_tab.to_csv(out / f"{args.tag}_seed{seed}_inner.csv", index=False)
            scores = {"out+text_lr": output_oof(D, outer, "out_lr", ("of", "ot")),
                      "out+text_hgb": output_oof(D, outer, "out_hgb", ("of", "ol", "ot")),
                      "out+text_pair": output_pair_oof(D, outer, ("of", "ot"))}
            chosen_all = {}
            if args.fixed:
                fx = outer_scores[args.fixed]
                ch = pd.DataFrame([{"fold": f, "recipe": args.fixed} for f in range(5)])
                scores["act[fixed]"] = fx
                scores["stacked[fixed]"] = stacked_ens_oof(D, ch, outer, fx, seed)
                scores["pair[fixed]"] = stacked_ens_oof(D, ch, outer, fx, seed, mode="pair")
            for crit, (sc, chosen) in res.items():
                scores[f"act[{crit}]"] = sc
                scores[f"stacked[{crit}]"] = stacked_ens_oof(D, chosen, outer, sc, seed)
                scores[f"pair[{crit}]"] = stacked_ens_oof(D, chosen, outer, sc, seed, mode="pair")
                chosen_all[crit] = chosen[["fold", "recipe", crit]].to_dict("records")
            np.savez_compressed(Path(args.cache) / f"scores_{args.tag}_seed{seed}.npz", outer=outer, **scores)
            per_recipe = {r: metrics(D, v, outer, np.isfinite(v), pairs) for r, v in outer_scores.items()}
            report = {"seed": seed, "recipes": recipes, "fixed": args.fixed, "chosen": chosen_all,
                      "per_recipe_unselected": per_recipe,
                      "comparison": compare(D, scores, outer, pairs, refs=("out+text_lr", "out+text_hgb", "out+text_pair"),
                                            n_boot=args.n_boot, seed=seed),
                      "secs": round(time.time() - t)}
            (out / f"{args.tag}_seed{seed}.json").write_text(json.dumps(report, indent=1))
            print(f"seed {seed} done in {report['secs']}s", flush=True)
            for k, v in report["comparison"]["scorers"].items():
                print(f"  {k:24s} " + "  ".join(f"{m} {v[m]['value']:.3f}" for m in
                                                ("auroc_all", "within_panel", "within_panel_pre", "rev_auc", "rev_auc_debate")), flush=True)
        return 0
    if args.cmd == "followup":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        D = Data(Path(args.cache))
        pairs = D.revision_pairs()
        ens = args.layer is None
        root, prefix = Path(D.meta["root"]), D.meta["prefix"]
        qp, sp = root / f"{prefix}-questions.jsonl", root / f"{prefix}-debate" / "agent_states.parquet"
        report = {"recipe": args.recipe, "layer": args.layer, "seeds": {}}
        letters = D.states.answer.to_numpy()

        def act(recipe, folds, excl=None):
            if ens:
                return ens_scores(D, recipe, folds, excl)
            position, kind, method = recipe.split(":")
            X, present = D.X(args.layer, position)
            return (run_config(D, X, present, letters, D.panels, kind, method, folds) if excl is None
                    else loco_masked(D, X, present, kind, method, folds, excl))

        def with_kind(recipe, extra):
            parts = recipe.split(":")
            parts[1] = f"{parts[1]}+{extra}"
            return ":".join(parts)

        for seed in args.seeds:
            folds = make_folds(D.groups, 5, seed)
            sc = {"act": act(args.recipe, folds)}
            for r in ("rtext", "rout"):
                sc[f"act[{r}]"] = act(with_kind(args.recipe, r), folds)
            chosen = pd.DataFrame([{"fold": f, "recipe": args.recipe, "layer": args.layer or 0} for f in range(5)])
            sc["stacked"] = (stacked_ens_oof(D, chosen, folds, sc["act"], seed) if ens
                             else stacked_oof(args.cache, D, chosen, folds, sc["act"], seed))
            sc["out+text_lr"] = output_oof(D, folds, "out_lr", ("of", "ot"))
            sc["out+text_hgb"] = output_oof(D, folds, "out_hgb", ("of", "ol", "ot"))
            rep = {"comparison": compare(D, sc, folds, pairs, refs=("out+text_lr", "out+text_hgb"),
                                         n_boot=args.n_boot, seed=seed)}
            rep["length_matched_within_panel"] = {k: matched_pairs(D, v, np.isfinite(v), qp, sp) for k, v in sc.items()}
            lo = {}
            for c in CONDS:
                post_c = ((D.states.condition == c) & (D.states.phase == "post")).to_numpy()
                sl = act(args.recipe, folds, post_c)
                pc = pairs[pairs.condition == c]
                for name, v in (("trained_with", sc["act"]), ("held_out", sl)):
                    d = v[pc.post.to_numpy()] - v[pc.pre.to_numpy()]
                    m = (D.states.condition == c).to_numpy() & np.isfinite(v)
                    lo.setdefault(c, {})[name] = {"rev_auc": auroc(pc.beneficial.to_numpy(), d),
                                                  "within_panel": pair_auroc(D.y[m], v[m], D.panels[m])[0]}
            rep["leave_condition_out"] = lo
            report["seeds"][str(seed)] = rep
            np.savez_compressed(Path(args.cache) / f"scores_{args.tag}_seed{seed}.npz", folds=folds, **sc)
            print(f"seed {seed}", flush=True)
            for k, v in rep["comparison"]["scorers"].items():
                print(f"  {k:14s} " + "  ".join(f"{m} {v[m]['value']:.3f}" for m in
                                                ("auroc_all", "within_panel", "within_panel_pre", "rev_auc", "rev_auc_debate")))
            print("  length-matched within-panel:", {k: round(v[0], 3) for k, v in rep["length_matched_within_panel"].items()},
                  "pairs", rep["length_matched_within_panel"]["act"][1])
            print("  leave-condition-out rev_auc (with, held out):",
                  {c: (round(v["trained_with"]["rev_auc"], 3), round(v["held_out"]["rev_auc"], 3)) for c, v in lo.items()})
        (out / f"{args.tag}.json").write_text(json.dumps(report, indent=1))
        return 0
    if args.cmd == "nested":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        D = Data(Path(args.cache))
        pairs = D.revision_pairs()
        for seed in args.seeds:
            t = time.time()
            res, outer, inner_tab = nested(args.cache, args.recipes, args.layers, args.criteria, seed, jobs=args.jobs)
            inner_tab.to_csv(out / f"{args.tag}_seed{seed}_inner.csv", index=False)
            scores = {"out_lr": output_oof(D, outer, "out_lr", ("of",)),
                      "out+text_lr": output_oof(D, outer, "out_lr", ("of", "ot")),
                      "out+text_hgb": output_oof(D, outer, "out_hgb", ("of", "ol", "ot"))}
            chosen_all = {}
            for crit, (sc, chosen) in res.items():
                scores[f"act[{crit}]"] = sc
                scores[f"stacked[{crit}]"] = stacked_oof(args.cache, D, chosen, outer, sc, seed, sets=("of", "ot"))
                chosen_all[crit] = chosen[["fold", "recipe", "layer", crit]].to_dict("records")
            np.savez_compressed(Path(args.cache) / f"scores_{args.tag}_seed{seed}.npz", outer=outer, **scores)
            report = {"seed": seed, "recipes": args.recipes, "layers": args.layers, "chosen": chosen_all,
                      "comparison": compare(D, scores, outer, pairs, refs=("out+text_lr", "out+text_hgb"),
                                            n_boot=args.n_boot, seed=seed),
                      "secs": round(time.time() - t)}
            (out / f"{args.tag}_seed{seed}.json").write_text(json.dumps(report, indent=1))
            print(f"seed {seed} done in {report['secs']}s", flush=True)
            for k, v in report["comparison"]["scorers"].items():
                line = "  ".join(f"{m} {v[m]['value']:.3f}" for m in ("auroc_all", "within_panel", "within_panel_pre", "rev_auc", "rev_auc_debate"))
                print(f"  {k:24s} {line}", flush=True)
        return 0
    if args.cmd == "cache":
        build_cache(Path(args.root), args.prefix, Path(args.cache), args.states_only)
        return 0
    if args.cmd == "screen":
        from joblib import Parallel, delayed
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        tasks = [(L, p) for p in args.positions for L in args.layers]
        t = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(screen_task)(args.cache, L, p, args.transforms, args.methods, args.seeds, args.k) for L, p in tasks)
        orows, oscores = output_rows(args.cache, args.seeds, args.k)
        rows = [r for rr, _ in res for r in rr] + orows
        scores = {k: v for _, sc in res for k, v in sc.items()} | oscores
        np.savez_compressed(Path(args.cache) / f"scores_{args.tag}.npz", **scores)
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{args.tag}.csv", index=False)
        print(f"{len(df)} rows in {time.time() - t:.0f}s -> {out / (args.tag + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
