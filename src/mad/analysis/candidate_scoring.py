"""Candidate scoring: fitted, batch-invariant correctness scorers and fold-safe metrics.

Written for the audit of the pilot probe search (docs/PROBE_AUDIT.md). Three rules, each a
correction of scripts/probe_search.py at 7787b68:

1. Every fitted quantity comes from TRAINING questions only: letter means, feature
   standardization, imputation values and which missingness indicators exist, PCA, model weights,
   and the normalizer that puts scores on a common scale. A fitted scorer scores each row from
   that row alone, so a row's score cannot change when other rows (unrelated questions, post
   states) are added to the evaluation batch. (The old ensemble and combination code
   standardized each held-out fold by its own mean and SD.)

2. Normalizers come from CROSS-FITTED reference scores inside the training set: the training
   questions are split into ``k_ref`` groups; a model fitted without group r scores group r; the
   mean and SD of those out-of-fold training scores define the normalizer. The scorer that is
   applied to evaluation rows is then refitted on ALL training rows and normalized with that
   normalizer. Alignment caveat: the refitted model sees (k_ref)/(k_ref-1) times as much data as
   each reference model, so its score scale can differ slightly (for a mean-difference direction
   the noise component of the direction shrinks with n). ``alignment`` diagnostics report the SD
   ratio of evaluation scores to reference scores; nothing is computed from evaluation scores.
   Platt calibrators are fitted on the same reference scores.

3. Metrics compare two scores only when one fitted model produced both: pairs inside a panel or
   a question (one question lies in one fold), and for revision AUROC, helpful vs harmful
   revisions inside one (condition, fold) cell. Cells are aggregated by pooling their
   comparison pairs (equivalently, AUROCs weighted by their pair counts). (The old rev_auc pooled
   score differences across fold models; rescaling one fold could change it.)

Bootstrap intervals resample questions with multiplicity weights: a within-question pair counts
w_q times; a pair of events from two questions counts w_a * w_b times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.calibrate import PlattScaler

POSITIONS = ("readout_last", "generation_last", "before_stated_answer", "stated_answer")
C_GRID = (0.001, 0.01, 0.1, 1.0)  # extended after a smoke run chose the 0.01 edge in every fold
OT_NAMES = ("ot_len", "ot_len_z", "ot_is_longest", "ot_n_longer", "ot_orig_pos", "ot_display_pos", "ot_n_options")


# ----------------------------------------------------------------------------- data


def make_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """A random partition of QUESTIONS into k folds (every state of a question in one fold).
    Identical to scripts/probe_search.make_folds, so seeds give the same partitions."""
    u = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(u)
    f = {g: i % k for i, g in enumerate(perm)}
    return np.array([f[g] for g in groups])


class StateData:
    """Answer-bearing states: round 0 once (condition ``round0``) plus every condition's revised
    states, with labels, output-only features and activation access by (layer, position).

    Conditions are read from the data, not hard-coded. Construction checks that no state key is
    duplicated and that round 0 appears exactly once per (question, agent)."""

    def __init__(self, states: pd.DataFrame, layer_fn=None, positions_present: dict | None = None):
        st = states.copy().reset_index(drop=True)
        st["question_id"] = st.question_id.astype(str)
        st.loc[st.phase == "pre", "condition"] = "round0"
        st["row_id"] = st.question_id + "|" + st.condition + "|" + st.phase + "|a" + st.agent_id.astype(str)
        dup = st.row_id.duplicated()
        if dup.any():
            raise ValueError(f"{int(dup.sum())} duplicate state keys, e.g. {st.row_id[dup].iloc[0]}")
        pre = st[st.phase == "pre"]
        if pre.groupby(["question_id", "agent_id"]).size().max() != 1:
            raise ValueError("round 0 appears more than once for some (question, agent)")
        self.states = st
        self.y = st.correct.to_numpy(int)
        self.qid = st.question_id.to_numpy()
        self.groups = pd.factorize(self.qid)[0]
        self.conditions = sorted(c for c in st.condition.unique() if c != "round0")
        self.panels = pd.factorize(st.question_id + "|" + st.condition + "|" + st.phase)[0]
        self.letters = st.answer.to_numpy()
        self._layer_fn = layer_fn            # (layer, position) -> (X aligned to states, present)
        self._present = positions_present or {}
        self._xcache: dict = {}

    # -- construction from the probe-search cache
    @classmethod
    def from_cache(cls, cache, conditions=None) -> StateData:
        cache = Path(cache)
        keys = pd.read_parquet(cache / "keys.parquet")
        st = pd.read_parquet(cache / "states.parquet")
        meta = json.loads((cache / "meta.json").read_text())
        st = st[st.phase.isin(["pre", "post"])].copy()
        st["question_id"] = st.question_id.astype(str)
        if conditions is not None:
            st = st[(st.phase == "pre") | st.condition.isin(conditions)]
        st = add_option_text(st, Path(meta["root"]), meta["prefix"])
        k = keys.copy()
        k["question_id"] = k.question_id.astype(str)
        k.loc[k.phase == "pre", "condition"] = "round0"
        kd = k.duplicated(["condition", "question_id", "agent_id", "phase", "position"])
        if kd.any():
            raise ValueError(f"{int(kd.sum())} duplicate activation keys in the cache")
        row = {(c, q, int(a), ph, pos): i for i, (c, q, a, ph, pos) in
               enumerate(k[["condition", "question_id", "agent_id", "phase", "position"]].itertuples(index=False))}
        st2 = st.copy()
        st2.loc[st2.phase == "pre", "condition"] = "round0"
        idx = {pos: np.array([row.get((c, q, int(a), ph, pos), -1) for c, q, a, ph in
                              st2[["condition", "question_id", "agent_id", "phase"]].itertuples(index=False)])
               for pos in POSITIONS}
        files: dict[int, np.ndarray] = {}

        def layer_fn(L, pos):
            if L not in files:
                files.clear()
                files[L] = np.load(cache / f"L{L:02d}.npy", mmap_mode="r")
            r = idx[pos]
            X = np.zeros((len(r), files[L].shape[1]), np.float32)
            X[r >= 0] = files[L][r[r >= 0]]
            return X, r >= 0

        D = cls(st, layer_fn, {pos: idx[pos] >= 0 for pos in POSITIONS})
        D.meta = meta
        return D

    def X(self, layer: int, position: str):
        key = (layer, position)
        if key not in self._xcache:
            if len(self._xcache) > 16:
                self._xcache.clear()
            self._xcache[key] = self._layer_fn(layer, position)
        return self._xcache[key]

    def present(self, position: str) -> np.ndarray:
        return self._present[position] if position in self._present else self.X(0, position)[1]

    def output(self, sets=("of", "ol", "ot")) -> tuple[np.ndarray, list[str]]:
        """Output-only features by set, with their exact column names:
        ``of`` letter-invariant output summaries, verbalized confidence, agreement, revised flag;
        ``ol`` per-letter option log-probabilities at each saved position;
        ``ot`` properties of the chosen option's text (no model involved)."""
        names = [c for c in self.states.columns if c[:3] in tuple(f"{s}_" for s in sets)]
        return self.states[names].to_numpy(float), names


def add_option_text(st: pd.DataFrame, root: Path, prefix: str) -> pd.DataFrame:
    """``ot_*`` columns: properties of the chosen option's text (see OT_NAMES)."""
    qs = {}
    for line in (root / f"{prefix}-questions.jsonl").read_text().splitlines():
        r = json.loads(line)
        qs[str(r["question_id"])] = r["options"]
    views = sorted(p for p in root.glob(f"{prefix}-*/agent_states.parquet") if "-c0" not in str(p.parent.name))
    ref = pd.read_parquet(views[0], columns=["question_id", "permutation"])
    perm = {str(q): [int(x) for x in p.split(",")] for q, p in zip(ref.question_id, ref.permutation, strict=True)}
    rows = []
    for q, a in zip(st.question_id.astype(str), st.answer, strict=True):
        opts, order = qs[q], perm[q]
        lens = np.array([len(o) for o in opts], float)
        orig = order[int(a)] if 0 <= int(a) < len(order) else 0
        L = lens[orig]
        rows.append([L, (L - lens.mean()) / (lens.std() + 1e-6), float(L == lens.max()),
                     float((lens > L).sum()), float(orig), float(a), float(len(opts))])
    st = st.copy()
    st[list(OT_NAMES)] = np.array(rows)
    return st


# ----------------------------------------------------------------------------- preprocessing


@dataclass
class Imputer:
    """Training means for missing values; missingness indicators only for columns that have
    missing values in TRAINING rows (a column missing only at evaluation time is imputed)."""

    means: np.ndarray = None
    ind_cols: np.ndarray = None

    def fit(self, M):
        with np.errstate(all="ignore"):
            mu = np.nanmean(M, 0)
        self.means = np.where(np.isfinite(mu), mu, 0.0)
        self.ind_cols = np.flatnonzero((~np.isfinite(M)).any(0))
        return self

    def transform(self, M):
        miss = ~np.isfinite(M)
        return np.hstack([np.where(miss, self.means, M), miss[:, self.ind_cols].astype(float)])


@dataclass
class Standardizer:
    mu: np.ndarray = None
    sd: np.ndarray = None

    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


@dataclass
class Normalizer:
    mu: float = 0.0
    sd: float = 1.0

    def fit(self, s):
        s = s[np.isfinite(s)]
        self.mu, self.sd = float(s.mean()), float(s.std() + 1e-9)
        return self

    def __call__(self, s):
        return (s - self.mu) / self.sd


# ----------------------------------------------------------------------------- activation probes


@dataclass
class LayerProbe:
    """One layer, one position. kind: 'raw' | 'letter' (subtract the training mean of states with
    the same answer letter), optionally '+rout' (first regress out output-only covariates, fitted
    on training rows). method: 'meandiff' | 'pca<k>' (PCA then logistic regression, C=1)."""

    kind: str
    method: str
    letter_means: dict = field(default_factory=dict)
    global_mean: np.ndarray = None
    resid: tuple = None
    std: Standardizer = None
    w: np.ndarray = None
    pca: object = None
    zstd: Standardizer = None
    lr: object = None

    def _prep(self, X, letters, C=None):
        X = X.astype(np.float32, copy=True)
        if self.resid is not None:
            imp, cstd, B = self.resid
            Z = np.hstack([np.ones((len(C), 1)), cstd.transform(imp.transform(C))])
            X -= (Z @ B).astype(np.float32)
        if "letter" in self.kind:
            for a in np.unique(letters):
                m = letters == a
                X[m] -= self.letter_means.get(int(a), self.global_mean)
        return X

    def fit(self, X, y, letters, C=None):
        if "rout" in self.kind:
            imp = Imputer().fit(C)
            Ci = imp.transform(C)
            cstd = Standardizer().fit(Ci)
            Z = np.hstack([np.ones((len(C), 1)), cstd.transform(Ci)])
            B, *_ = np.linalg.lstsq(Z, X, rcond=None)
            self.resid = (imp, cstd, B)
        if "letter" in self.kind:
            Xr = self._prep_resid_only(X, C)
            self.global_mean = Xr.mean(0)
            for a in np.unique(letters):
                m = letters == a
                if m.sum() >= 5:
                    self.letter_means[int(a)] = Xr[m].mean(0)
        Xp = self._prep(X, letters, C)
        self.std = Standardizer().fit(Xp)
        Xs = self.std.transform(Xp)
        if self.method == "meandiff":
            self.w = Xs[y == 1].mean(0) - Xs[y == 0].mean(0)
        elif self.method.startswith("pca"):
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            k = int(self.method[3:])
            self.pca = PCA(n_components=min(k, len(Xs) - 1), svd_solver="randomized", random_state=0).fit(Xs)
            Zp = self.pca.transform(Xs)
            self.zstd = Standardizer().fit(Zp)
            self.lr = LogisticRegression(C=1.0, max_iter=4000).fit(self.zstd.transform(Zp), y)
        else:
            raise ValueError(self.method)
        return self

    def _prep_resid_only(self, X, C):
        if self.resid is None:
            return X.astype(np.float32, copy=True)
        imp, cstd, B = self.resid
        Z = np.hstack([np.ones((len(C), 1)), cstd.transform(imp.transform(C))])
        return X.astype(np.float32) - (Z @ B).astype(np.float32)

    def score(self, X, letters, C=None):
        Xs = self.std.transform(self._prep(X, letters, C))
        if self.method == "meandiff":
            return Xs @ self.w
        return self.lr.decision_function(self.zstd.transform(self.pca.transform(Xs)))


def parse_recipe(recipe: str):
    """'position:kind:method:a-b[:step]' -> (position, kind, method, layers). Default step 3."""
    parts = recipe.split(":")
    a, b = (int(x) for x in parts[3].split("-"))
    step = int(parts[4]) if len(parts) > 4 else 3
    return parts[0], parts[1], parts[2], list(range(a, b + 1, step))


def _ref_folds(D: StateData, T: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Reference folds over TRAINING questions only (-1 elsewhere)."""
    f = np.full(len(D.y), -1)
    f[T] = make_folds(D.groups[T], k, seed)
    return f


class ActivationScorer:
    """Layer-band ensemble: per layer a LayerProbe; per layer a normalizer from cross-fitted
    training reference scores; the ensemble is the mean of normalized layer scores, normalized
    again from the ensemble's reference scores; Platt calibration on the same references."""

    def __init__(self, recipe: str, k_ref: int = 5, seed: int = 0, cov_sets=("of", "ol", "ot")):
        self.recipe = recipe
        self.position, self.kind, self.method, self.layers = parse_recipe(recipe)
        self.k_ref, self.seed, self.cov_sets = k_ref, seed, cov_sets

    def _cov(self, D):
        return D.output(self.cov_sets)[0] if "rout" in self.kind else None

    def fit(self, D: StateData, train: np.ndarray):
        C = self._cov(D)
        present = D.present(self.position)
        T = train & present
        rf = _ref_folds(D, T, self.k_ref, self.seed)
        self.probes, self.layer_norm, refs = {}, {}, []
        for L in self.layers:
            X, _ = D.X(L, self.position)
            ref = np.full(len(D.y), np.nan)
            for r in range(self.k_ref):
                a, b = T & (rf != r), T & (rf == r)
                p = LayerProbe(self.kind, self.method).fit(X[a], D.y[a], D.letters[a], None if C is None else C[a])
                ref[b] = p.score(X[b], D.letters[b], None if C is None else C[b])
            self.layer_norm[L] = Normalizer().fit(ref[T])
            refs.append(self.layer_norm[L](ref))
            self.probes[L] = LayerProbe(self.kind, self.method).fit(X[T], D.y[T], D.letters[T],
                                                                     None if C is None else C[T])
        ens_ref = np.mean(refs, axis=0)
        self.norm = Normalizer().fit(ens_ref[T])
        self.ref_z = self.norm(ens_ref)               # NaN outside training rows
        self.platt = PlattScaler().fit(self.ref_z[T], D.y[T])
        self.train = T
        return self

    def score(self, D: StateData, rows: np.ndarray):
        """(raw, z, p) for ``rows``; NaN where the state is absent. Row-wise: no batch statistics."""
        C = self._cov(D)
        present = D.present(self.position)
        m = rows & present
        acc = np.zeros(int(m.sum()))
        for L in self.layers:
            X, _ = D.X(L, self.position)
            acc += self.layer_norm[L](self.probes[L].score(X[m], D.letters[m], None if C is None else C[m]))
        raw = np.full(len(D.y), np.nan)
        raw[m] = acc / len(self.layers)
        z = self.norm(raw)
        p = np.full(len(D.y), np.nan)
        p[m] = self.platt.transform(z[m])
        return raw, z, p

    def params(self) -> dict:
        return {"recipe": self.recipe, "position": self.position, "kind": self.kind, "method": self.method,
                "layers": self.layers, "k_ref": self.k_ref,
                "layer_normalizers": {int(L): [n.mu, n.sd] for L, n in self.layer_norm.items()},
                "ensemble_normalizer": [self.norm.mu, self.norm.sd],
                "platt": _platt_params(self.platt), "n_train_states": int(self.train.sum())}


def _platt_params(p: PlattScaler) -> dict:
    out = {"constant": None if getattr(p, "constant", None) is None else float(p.constant)}
    lr = getattr(p, "_lr", None)
    if out["constant"] is None and lr is not None and hasattr(lr, "coef_"):
        out.update({"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])})
    return out


# ----------------------------------------------------------------------------- output-only and combined


def _lr_cv_logloss(Z, y, g, C, k=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    f = make_folds(g, k, seed)
    ll = []
    for r in range(k):
        a, b = f != r, f == r
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        m = LogisticRegression(C=C, max_iter=4000).fit(Z[a], y[a])
        ll.append(log_loss(y[b], m.predict_proba(Z[b])[:, 1], labels=[0, 1]))
    return float(np.mean(ll))


class TabularScorer:
    """Logistic regression (C fixed or chosen from C_GRID by grouped inner-CV log-loss on the
    training rows) or gradient boosting, on a named feature matrix. Imputation, indicators and
    standardization are fitted on training rows. Normalizer and Platt from cross-fitted training
    reference scores. ``features(D, rows)`` returns the matrix; subclasses add columns."""

    def __init__(self, name, sets=("of", "ol", "ot"), model="lr", C=None, k_ref=5, seed=0):
        self.name, self.sets, self.model, self.C_fixed, self.k_ref, self.seed = name, sets, model, C, k_ref, seed

    def features(self, D):
        return D.output(self.sets)

    def _fit_model(self, Z, y, C):
        if self.model == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0).fit(Z, y)
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=C, max_iter=4000).fit(Z, y)

    def _decision(self, m, Z):
        if self.model == "hgb":
            p = np.clip(m.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        return m.decision_function(Z)

    def _prep_fit(self, M, rows):
        imp = Imputer().fit(M[rows])
        Mi = imp.transform(M)
        std = Standardizer().fit(Mi[rows]) if self.model == "lr" else None
        return imp, std

    def _prep(self, M, imp, std):
        Mi = imp.transform(M)
        return std.transform(Mi) if std is not None else Mi

    def fit(self, D: StateData, train: np.ndarray):
        M, self.names = self.features(D)
        T = train & np.isfinite(M).any(1)
        if self.model == "lr" and self.C_fixed is None:
            imp, std = self._prep_fit(M, T)
            Z = self._prep(M, imp, std)
            self.cv = {C: _lr_cv_logloss(Z[T], D.y[T], D.groups[T], C, seed=self.seed) for C in C_GRID}
            self.C = min(self.cv, key=self.cv.get)
        else:
            self.C, self.cv = (self.C_fixed if self.C_fixed is not None else 1.0), None
        rf = _ref_folds(D, T, self.k_ref, self.seed)
        ref = np.full(len(D.y), np.nan)
        for r in range(self.k_ref):
            a, b = T & (rf != r), T & (rf == r)
            imp, std = self._prep_fit(M, a)
            m = self._fit_model(self._prep(M[a], imp, std), D.y[a], self.C)
            ref[b] = self._decision(m, self._prep(M[b], imp, std))
        self.norm = Normalizer().fit(ref[T])
        self.ref_z = self.norm(ref)
        self.platt = PlattScaler().fit(self.ref_z[T], D.y[T])
        self.imp, self.std = self._prep_fit(M, T)
        self.m = self._fit_model(self._prep(M[T], self.imp, self.std), D.y[T], self.C)
        self.train = T
        return self

    def score(self, D: StateData, rows: np.ndarray):
        M, _ = self.features(D)
        raw = np.full(len(D.y), np.nan)
        raw[rows] = self._decision(self.m, self._prep(M[rows], self.imp, self.std))
        z = self.norm(raw)
        p = np.full(len(D.y), np.nan)
        p[rows] = self.platt.transform(z[rows])
        return raw, z, p

    def params(self) -> dict:
        out = {"name": self.name, "model": self.model, "feature_names": list(self.names), "C": self.C,
               "C_cv_logloss": self.cv, "normalizer": [self.norm.mu, self.norm.sd],
               "platt": _platt_params(self.platt), "n_train_states": int(self.train.sum())}
        if self.model == "lr":
            out["coef"] = [float(c) for c in self.m.coef_[0]]
            out["intercept"] = float(self.m.intercept_[0])
        return out


class CombinedScorer(TabularScorer):
    """The matched output model plus one activation column. Training rows use the activation
    scorer's cross-fitted REFERENCE z (never in-sample scores); evaluation rows use its fitted z.
    Same feature set, same C grid and same selection rule as the matched output-only scorer."""

    def __init__(self, name, act, sets=("of", "ol", "ot"), C=None, k_ref=5, seed=0, fallback=None):
        super().__init__(name, sets, "lr", C, k_ref, seed)
        self.act, self.fallback = act, fallback
        self._eval_z = None

    def features(self, D):
        M, names = D.output(self.sets)
        a = self.act.ref_z.copy() if self._eval_z is None else self._eval_z
        if self.fallback is not None:  # operational fallback for absent activation states
            fb = self.fallback.ref_z if self._eval_z is None else self._fb_eval
            a = np.where(np.isfinite(a), a, fb)
        return np.hstack([M, a[:, None]]), [*names, f"act:{self.act.recipe}"]

    def score(self, D, rows):
        self._eval_z = self.act.score(D, rows)[1]
        if self.fallback is not None:
            self._fb_eval = self.fallback.score(D, rows)[1]
        try:
            return super().score(D, rows)
        finally:
            self._eval_z = None


class SumScorer:
    """Equal-weight sum of two scorers' training-normalized z scores (post hoc rule; no fitting
    beyond the components). Platt on the sum of their reference z."""

    def __init__(self, name, a, b, fallback_for_a=None):
        self.name, self.a, self.b, self.fb = name, a, b, fallback_for_a

    def fit(self, D, train):
        ra = self.a.ref_z if self.fb is None else np.where(np.isfinite(self.a.ref_z), self.a.ref_z, self.fb.ref_z)
        self.ref_z = ra + self.b.ref_z
        T = train & np.isfinite(self.ref_z)
        self.platt = PlattScaler().fit(self.ref_z[T], D.y[T])
        self.train = T
        return self

    def score(self, D, rows):
        za = self.a.score(D, rows)[1]
        if self.fb is not None:
            za = np.where(np.isfinite(za), za, self.fb.score(D, rows)[1])
        z = za + self.b.score(D, rows)[1]
        p = np.full(len(D.y), np.nan)
        ok = rows & np.isfinite(z)
        p[ok] = self.platt.transform(z[ok])
        return z, z, p

    def params(self):
        return {"name": self.name, "components": [getattr(self.a, "recipe", getattr(self.a, "name", "")),
                                                  self.b.name], "platt": _platt_params(self.platt)}


class FallbackScorer:
    """Activation z where the state exists, else the given scorer's z (all-question operational
    fallback for positions that are missing for some states)."""

    def __init__(self, name, act, fb):
        self.name, self.act, self.fb = name, act, fb

    def fit(self, D, train):
        self.ref_z = np.where(np.isfinite(self.act.ref_z), self.act.ref_z, self.fb.ref_z)
        T = train & np.isfinite(self.ref_z)
        self.platt = PlattScaler().fit(self.ref_z[T], D.y[T])
        return self

    def score(self, D, rows):
        za, zb = self.act.score(D, rows)[1], self.fb.score(D, rows)[1]
        z = np.where(np.isfinite(za), za, zb)
        p = np.full(len(D.y), np.nan)
        ok = rows & np.isfinite(z)
        p[ok] = self.platt.transform(z[ok])
        return z, z, p

    def params(self):
        return {"name": self.name, "activation": self.act.recipe, "fallback": self.fb.name}


# ----------------------------------------------------------------------------- metrics


def pairs_within(groups: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """All (correct, incorrect) index pairs inside each group, among ``mask`` rows."""
    good, bad = [], []
    idx_all = np.flatnonzero(mask)
    order = np.argsort(groups[idx_all], kind="stable")
    idx_all = idx_all[order]
    gs = groups[idx_all]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], len(idx_all)]
    for s, e in zip(starts, ends, strict=True):
        idx = idx_all[s:e]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(pos) and len(neg):
            good.append(np.repeat(pos, len(neg)))
            bad.append(np.tile(neg, len(pos)))
    if not good:
        return np.array([], int), np.array([], int)
    return np.concatenate(good), np.concatenate(bad)


def concordance(s, i, j):
    """1 where s[i] > s[j], 0.5 for ties, 0 otherwise."""
    d = s[i] - s[j]
    return (d > 0) + 0.5 * (d == 0)


def revision_events(D: StateData, condition: str) -> pd.DataFrame:
    """Every agent's (first, revised) pair in a condition, with its type: helpful (wrong ->
    right), harmful (right -> wrong), wrong_to_wrong (changed, both wrong), unchanged."""
    st = D.states.reset_index()
    pre = st[st.phase == "pre"].set_index(["question_id", "agent_id"])
    out = []
    for _, r in st[(st.phase == "post") & (st.condition == condition)].iterrows():
        p = pre.loc[(r.question_id, r.agent_id)]
        if p.answer == r.answer:
            kind = "unchanged"
        elif r.correct and not p.correct:
            kind = "helpful"
        elif p.correct and not r.correct:
            kind = "harmful"
        else:
            kind = "wrong_to_wrong"
        out.append({"pre": int(p["index"]), "post": int(r["index"]), "question_id": r.question_id,
                    "agent_id": int(r.agent_id), "kind": kind})
    return pd.DataFrame(out)


def revision_pairs(ev: pd.DataFrame, folds: np.ndarray, D: StateData):
    """(helpful event, harmful event) pairs inside one fold cell -- both differences come from
    the same fitted model. Returns pair index arrays into ``ev`` and per-cell counts."""
    hi, hj, cells = [], [], []
    fold_of = folds[ev.post.to_numpy()]
    for f in np.unique(fold_of):
        h = np.flatnonzero((fold_of == f) & (ev.kind == "helpful").to_numpy())
        x = np.flatnonzero((fold_of == f) & (ev.kind == "harmful").to_numpy())
        cells.append({"fold": int(f), "helpful": int(len(h)), "harmful": int(len(x)),
                      "pairs": int(len(h) * len(x)), "one_class": bool(len(h) == 0 or len(x) == 0)})
        if len(h) and len(x):
            hi.append(np.repeat(h, len(x)))
            hj.append(np.tile(x, len(h)))
    if not hi:
        return np.array([], int), np.array([], int), cells
    return np.concatenate(hi), np.concatenate(hj), cells


def weighted_rate(values, weights):
    """Sum(w * v) / Sum(w) for each row of ``weights`` (shape n_boot x n)."""
    den = weights.sum(1)
    return np.where(den > 0, weights @ values / np.where(den > 0, den, 1), np.nan)


def boot_weights(n_questions: int, n_boot: int, seed: int) -> np.ndarray:
    """Row 0 is the point estimate (all ones); rows 1.. are question multiplicities."""
    rng = np.random.default_rng(seed)
    W = [np.ones(n_questions)]
    W += [np.bincount(rng.integers(0, n_questions, n_questions), minlength=n_questions) for _ in range(n_boot)]
    return np.vstack(W).astype(float)


def summarize(vals: np.ndarray) -> dict:
    return {"value": float(vals[0]), "ci95": [float(np.nanquantile(vals[1:], 0.025)),
                                              float(np.nanquantile(vals[1:], 0.975))]}


def population_masks(D: StateData, condition: str) -> dict:
    """State sets for one condition: its round-0 states, its revised states, and both (the six
    candidates per question)."""
    pre = (D.states.phase == "pre").to_numpy()
    post = ((D.states.phase == "post") & (D.states.condition == condition)).to_numpy()
    return {"pre": pre, "post": post, "six": pre | post}


def eligible_questions(D: StateData, present: np.ndarray, condition: str | None) -> np.ndarray:
    """Questions whose round-0 states (and, for a condition, its revised states) are all present."""
    m = population_masks(D, condition)["six"] if condition else (D.states.phase == "pre").to_numpy()
    st = D.states.assign(_p=present)[m]
    ok = st.groupby("question_id")._p.all()
    return ok[ok].index.to_numpy()


def evaluate_scores(D: StateData, z: np.ndarray, folds: np.ndarray, condition: str, questions: np.ndarray,
                    W: np.ndarray, qindex: dict, return_vals: bool = False) -> dict:
    """Fold-safe ranking metrics for one condition on ``questions`` (all states present), each
    with question-bootstrap values (row 0 of ``W`` is the point estimate).

    state_auc_post    revised states; correct vs incorrect pairs within a fold, pooled over folds
    state_auc_pre     round-0 states, likewise
    panel_pre         pairs within each round-0 panel (three first answers)
    panel_post        pairs within each revised panel
    six               pairs among the six candidates of a question (three first + three revised)
    revision_auc      helpful vs harmful: score(revised) - score(first), pairs inside (fold) cells
    """
    inq = np.isin(D.qid, questions)
    masks = population_masks(D, condition)
    qi = np.array([qindex[q] for q in D.qid])
    out, counts, raw = {}, {}, {}

    def within(name, mask, groups):
        i, j = pairs_within(groups, D.y, mask & inq & np.isfinite(z))
        c = concordance(z, i, j)
        vals = weighted_rate(c, W[:, qi[i]]) if len(i) else np.full(len(W), np.nan)
        out[name], raw[name] = summarize(vals), vals
        counts[name] = {"pairs": int(len(i)), "questions": int(len(np.unique(D.qid[i]))) if len(i) else 0}

    def across(name, mask):
        i, j = pairs_within(folds, D.y, mask & inq & np.isfinite(z))
        c = concordance(z, i, j)
        vals = weighted_rate(c, W[:, qi[i]] * W[:, qi[j]]) if len(i) else np.full(len(W), np.nan)
        out[name], raw[name] = summarize(vals), vals
        counts[name] = {"pairs": int(len(i)), "states": int((mask & inq & np.isfinite(z)).sum())}

    across("state_auc_post", masks["post"])
    across("state_auc_pre", masks["pre"])
    within("panel_pre", masks["pre"], D.panels)
    within("panel_post", masks["post"], D.panels)
    within("six", masks["six"], D.groups)
    ev = revision_events(D, condition)
    ev = ev[np.isin(ev.question_id, questions)].reset_index(drop=True)
    d = z[ev.post.to_numpy()] - z[ev.pre.to_numpy()]
    ok = np.isfinite(d)
    ev, d = ev[ok].reset_index(drop=True), d[ok]
    i, j, cells = revision_pairs(ev, folds, D)
    c = (d[i] > d[j]) + 0.5 * (d[i] == d[j])
    eq = np.array([qindex[q] for q in ev.question_id])
    vals = weighted_rate(c, W[:, eq[i]] * W[:, eq[j]]) if len(i) else np.full(len(W), np.nan)
    out["revision_auc"], raw["revision_auc"] = summarize(vals), vals
    counts["revision_auc"] = {"pairs": int(len(i)), "cells": cells,
                              "helpful": int((ev.kind == "helpful").sum()), "harmful": int((ev.kind == "harmful").sum()),
                              "wrong_to_wrong": int((ev.kind == "wrong_to_wrong").sum()),
                              "questions": int(len(np.unique(np.r_[ev.question_id[i], ev.question_id[j]]))) if len(i) else 0}
    w2w = ev.kind == "wrong_to_wrong"
    counts["wrong_to_wrong_rated_higher_after"] = float((d[w2w.to_numpy()] > 0).mean()) if w2w.any() else None
    res = {"metrics": out, "counts": counts}
    if return_vals:
        res["vals"] = raw
    return res


def calibration(D, p, rows, base_rate) -> dict:
    """Brier, log-loss and ECE (10 bins) of Platt probabilities vs a constant training rate."""
    m = rows & np.isfinite(p)
    y, q = D.y[m], np.clip(p[m], 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    c = np.clip(base_rate, 1e-6, 1 - 1e-6)
    bins = np.minimum((q * 10).astype(int), 9)
    ece = float(sum(abs(y[bins == b].mean() - q[bins == b].mean()) * (bins == b).mean() for b in np.unique(bins)))
    return {"n": int(m.sum()), "brier": float(np.mean((q - y) ** 2)), "logloss": ll, "ece": ece,
            "constant_brier": float(np.mean((c - y) ** 2)),
            "constant_logloss": float(-np.mean(y * np.log(c) + (1 - y) * np.log(1 - c)))}


def _vote(answers, fallback, tie):
    from mad.analysis.labels import panel_vote
    return panel_vote(answers, [0.0] * len(answers), fallback, tie)


def _random_tie_expectation(answers, gold):
    """Expected correctness of a plurality vote whose ties are broken uniformly at random
    (label-independent tie-break sensitivity)."""
    from collections import Counter
    c = Counter(answers)
    top = max(c.values())
    tied = [a for a, k in c.items() if k == top]
    return (1.0 / len(tied)) if gold in tied else 0.0


def _change_kind(a, b, gold) -> str:
    if a == b:
        return "unchanged"
    if b == gold:
        return "helpful"
    if a == gold:
        return "harmful"
    return "wrong_to_wrong"


def decisions(D: StateData, condition: str, scores: dict, taus: dict | None = None,
              tie: str = "round0_then_agent0") -> pd.DataFrame:
    """Per question: answers chosen by every rule, whether they are correct, and where they came
    from. ``scores``: name -> z (higher = more likely correct). Absent scores never win a
    selection; a panel with no scores falls back to the original vote."""
    st = D.states.reset_index(drop=True)
    pre = st[st.phase == "pre"]
    post = st[(st.phase == "post") & (st.condition == condition)]
    rows = []
    for q, g0 in pre.groupby("question_id", sort=True):
        g0 = g0.sort_values("agent_id")
        g1 = post[post.question_id == q].sort_values("agent_id")
        if len(g1) != len(g0):
            continue
        gold = int(g0.gold_display.iloc[0])
        a0, a1 = [int(x) for x in g0.answer], [int(x) for x in g1.answer]
        i0, i1 = g0.index.to_numpy(), g1.index.to_numpy()
        orig = _vote(a0, None, tie)
        r = {"question_id": q, "gold": gold, "round0_answers": ",".join(map(str, a0)),
             "revised_answers": ",".join(map(str, a1)),
             "any_round0_correct": int(gold in a0), "any_revised_correct": int(gold in a1),
             "debate_introduced_correct": int(gold in a1 and gold not in a0),
             "original_vote": orig, "revised_vote": _vote(a1, a0, tie), "pooled6_vote": _vote(a0 + a1, a0, tie),
             "original_vote_random_tie": _random_tie_expectation(a0, gold),
             "revised_vote_random_tie": _random_tie_expectation(a1, gold),
             "pooled6_vote_random_tie": _random_tie_expectation(a0 + a1, gold)}
        for name, z in scores.items():
            s0, s1 = z[i0], z[i1]
            if np.isfinite(s0).any():
                r[f"best_original[{name}]"] = a0[int(np.nanargmax(s0))]
            else:
                r[f"best_original[{name}]"] = orig
            s6 = np.r_[s0, s1]
            if np.isfinite(s6).any():
                k = int(np.nanargmax(s6))
                pick = (a0 + a1)[k]
                r[f"best_six[{name}]"] = pick
                r[f"best_six_origin[{name}]"] = "round0" if k < len(a0) else "revision"
                r[f"best_six_new_option[{name}]"] = int(pick not in a0)
            else:
                r[f"best_six[{name}]"], r[f"best_six_origin[{name}]"], r[f"best_six_new_option[{name}]"] = orig, "vote", 0
            for gname, tau in [("gate0", 0.0)] + ([("gate_tuned", taus[name])] if taus and name in taus else []):
                acc = [bool(np.isfinite(x) and np.isfinite(y) and (y - x) > tau) for x, y in zip(s0, s1, strict=True)]
                fin = [b if ok else a for a, b, ok in zip(a0, a1, acc, strict=True)]
                r[f"{gname}[{name}]"] = _vote(fin, a0, tie)
                for kind in ("helpful", "harmful", "wrong_to_wrong"):
                    ks = [_change_kind(a, b, gold) == kind for a, b in zip(a0, a1, strict=True)]
                    r[f"{gname}_{kind}_accepted[{name}]"] = int(sum(k and o for k, o in zip(ks, acc, strict=True)))
                    r[f"{gname}_{kind}_blocked[{name}]"] = int(sum(k and not o for k, o in zip(ks, acc, strict=True)))
        rows.append(r)
    df = pd.DataFrame(rows).set_index("question_id")
    for c in [c for c in df.columns if c.endswith("_vote") or c.startswith(("best_original[", "best_six[", "gate0[", "gate_tuned["))]:
        df[f"correct:{c}"] = (df[c] == df.gold).astype(int)
    return df


TAU_GRID = tuple(float(x) for x in np.round(np.arange(-1.5, 1.51, 0.1), 2))


def tune_gate_tau(D: StateData, condition: str, ref_z: np.ndarray, train_questions, tie="round0_then_agent0",
                  grid=TAU_GRID) -> float:
    """One validation-tuned threshold: accept a revision iff z(revised) - z(first) > tau, with tau
    maximizing the panel-correct count over TRAINING questions using cross-fitted reference z.
    Ties go to the tau nearest 0."""
    st = D.states.reset_index(drop=True)
    pre = st[(st.phase == "pre") & st.question_id.isin(train_questions)]
    post = st[(st.phase == "post") & (st.condition == condition) & st.question_id.isin(train_questions)]
    groups = []
    for q, g0 in pre.groupby("question_id"):
        g0 = g0.sort_values("agent_id")
        g1 = post[post.question_id == q].sort_values("agent_id")
        if len(g1) == len(g0):
            groups.append((int(g0.gold_display.iloc[0]), [int(x) for x in g0.answer], [int(x) for x in g1.answer],
                           ref_z[g0.index.to_numpy()], ref_z[g1.index.to_numpy()]))
    best = None
    for tau in sorted(grid, key=abs):
        n = 0
        for gold, a0, a1, s0, s1 in groups:
            fin = [b if (np.isfinite(x) and np.isfinite(y) and (y - x) > tau) else a
                   for a, b, x, y in zip(a0, a1, s0, s1, strict=True)]
            n += int(_vote(fin, a0, tie) == gold)
        if best is None or n > best[1]:
            best = (float(tau), n)
    return best[0]
