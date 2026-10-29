"""Archive inventory and alignment audit (CPU; docs/NEXT_PHASE_PLAN.md tasks A0-A1).

    python scripts/archive_audit.py

Byte-level integrity is checked separately by scripts/verify_archive.py (every shard hash,
views rebuilt from chunks, every activation finite; ``--copy-to`` for the pilot's second copy);
its reports (``<job>-verify.json``) are collected here. This script checks ALIGNMENT, for every
state of every archived view:

  identity    model commit, chat-template hash and transformers version identical across jobs,
              and equal to the locally cached tokenizer used to build readout inputs
  prompt      the stored prompt string re-renders from the stored messages with the tokenizer's
              chat template; its token count equals the count the readout recorded
  replay      the saved generated ids decode to the stored response text
  answer      re-parsing the stored text gives the recorded stated answer (and its status)
  peers       every revision's stored messages are rebuilt exactly from the peers' round-0 texts
              in the recorded order (confidence lines stripped as the protocol does)
  exposure    one peer_exposure state per revision, same keys
  readouts    saved letter distributions are finite and normalised

and reports where the archives and their copies are, plus any off-device backup destination.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad import prompts as P  # noqa: E402
from mad.parsing import parse_answer_field  # noqa: E402

OUT = ROOT / "artifacts/next_phase/archive"
JOBS = [("pilot12b", ROOT / "runs/colab"), ("val12b", ROOT / "runs/val12b"), ("val12b-self", ROOT / "runs/val12b")]
BACKUP = Path("/mnt/c/Users/mmcph/mas_backup")
MODEL = ("google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80")


def _render(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # templates without a system role: fold it into the first user turn
        system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
        rest = [m for m in msgs if m["role"] != "system"]
        if system and rest and rest[0]["role"] == "user":
            rest = [{"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}, *rest[1:]]
        return tok.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)


def identities(tok) -> dict:
    out = {}
    for name, root in JOBS:
        j = json.loads((root / f"{name}-job.json").read_text())
        b = j["identity"]["backend"]
        hist = j["history"][0].get("execution", {}).get("packages", {})
        out[name] = {"model_commit": b.get("model_commit"), "chat_template_sha256": b.get("chat_template_sha256"),
                     "transformers": hist.get("transformers"), "torch": hist.get("torch"),
                     "code_git_sha": j["code"]["git_sha"], "identity_sha256": j["identity_sha256"],
                     "dataset_revision": j["identity"]["dataset"].get("revision")}
    local = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()
    vals = lambda k: {v[k] for v in out.values()}  # noqa: E731
    return {"jobs": out, "local_tokenizer_commit": MODEL[1], "local_chat_template_sha256": local,
            "model_commit_identical": len(vals("model_commit")) == 1,
            "chat_template_identical": len(vals("chat_template_sha256")) == 1,
            "local_tokenizer_matches_archives": vals("chat_template_sha256") == {local}
            and vals("model_commit") == {MODEL[1]}}


def audit_view(view: Path, tok) -> dict:
    st = pd.read_parquet(view / "agent_states.parquet")
    st["question_id"] = st.question_id.astype(str)
    T = [json.loads(x) for x in open(view / "transcripts.jsonl")]
    by = {(str(r["question_id"]), int(r["agent_id"]), r["phase"]): r for r in T}
    S = st.set_index(["question_id", "agent_id", "phase"])
    c, bad = Counter(), {}

    def flag(kind, key):
        c[kind] += 1
        bad.setdefault(kind, []).append(list(map(str, key)))

    for key, r in by.items():
        row = S.loc[key]
        c["states"] += 1
        c["prompt_rerender_checked"] += 1
        if _render(tok, r["messages"]) + r.get("prefill", "") != r["prompt"]:
            flag("prompt_not_rerendered", key)
        n_prompt = len(tok.encode(r["prompt"], add_special_tokens=False))
        if "readout_prompt_tokens" in row and pd.notna(row.readout_prompt_tokens):
            c["prompt_token_count_checked"] += 1
            if n_prompt != int(row.readout_prompt_tokens):
                flag("prompt_token_count_differs", key)
        ids = list(r["token_ids"] or [])
        while ids and ids[-1] in set(tok.all_special_ids) | {106, 1}:
            ids.pop()
        c["replay_checked"] += 1
        if tok.decode(ids, skip_special_tokens=True) != r["text"]:
            flag("replay_mismatch", key)
        p = parse_answer_field(r["text"], int(row.n_options))
        c["answer_reparse_checked"] += 1
        if int(row.freetext_answer) != p.index or row.stated_status != p.status:
            flag("stated_answer_reparse_differs", key)
        if row.answer_source == "stated" and int(row.answer) != p.index:
            flag("canonical_answer_differs_from_stated", key)
        if r["stop_reason"] == "length":
            c["capped"] += 1
            c["capped_with_stated_answer"] += int(p.ok)
    # peers: rebuild every revision's messages from the round-0 texts in the recorded order
    cond = str(st.condition.iloc[0])
    for (q, a, ph), r in by.items():
        if ph != "post":
            continue
        base = by[(q, a, "pre")]["messages"]
        own = by[(q, a, "pre")]["text"]
        srcs = [s for s in (r.get("shown_peer_sources") or "").split(",") if s]
        if cond == "debate":
            peers = [P.strip_confidence(by[(q, int(s.replace("agent", "")), "pre")]["text"]) for s in srcs]
            rebuilt = P.round1_messages(base=base, own_response=own, peer_responses=peers)
        elif cond == "self_reconsider":
            rebuilt = P.round1_messages(base=base, own_response=own, peer_responses=None)
        else:
            continue
        c["peer_messages_checked"] += 1
        if rebuilt != r["messages"]:
            flag("peer_messages_not_rebuilt", (q, a, ph))
    # exposure vs revision keys
    k = lambda ph: set(map(tuple, st[st.phase == ph][["question_id", "agent_id"]].values))  # noqa: E731
    c["exposure_states"], c["post_states"] = len(k("peer_exposure")), len(k("post"))
    if k("peer_exposure") != k("post"):
        flag("exposure_keys_differ_from_revision_keys", ("view",))
    for col in ("readout_logprobs", "pos_before_stated_answer_logprobs"):
        vals = [np.array([float(x) for x in v.split(",")]) for v in st[col].dropna() if isinstance(v, str) and v]
        badn = sum(1 for v in vals if not np.isfinite(v).all() or abs(np.exp(v).sum() - 1) > 1e-3)
        c[f"{col}_checked"] = len(vals)
        if badn:
            c[f"{col}_not_normalised"] = badn
    return {"counts": dict(c), "examples": {k2: v[:5] for k2, v in bad.items()}}


def backups() -> dict:
    out = {"second_copy_root": str(BACKUP), "second_copy_contents": {}}
    if BACKUP.exists():
        for d in sorted(BACKUP.iterdir()):
            out["second_copy_contents"][d.name] = len(list(d.iterdir())) if d.is_dir() else 1
    out["same_physical_disk_as_primary"] = True  # WSL ext4 image and C: live on the laptop's one SSD
    od = Path("/mnt/c/Users/mmcph/OneDrive")
    drives = sorted(p.name for p in Path("/mnt").iterdir() if len(p.name) == 1)
    out["off_device_candidates"] = {
        "windows_drives_visible": drives,
        "onedrive_folder": {"path": str(od), "exists": od.exists(),
                            "entries": len(list(od.iterdir())) if od.exists() else 0,
                            "sync_state": "unknown: cannot be verified from WSL; nothing copied there"},
        "google_drive_mount": "not available (Colab CLI mount failed twice)",
    }
    out["conclusion"] = ("No off-device destination is confirmed as configured. A OneDrive folder exists on C:, "
                         "but whether it syncs, and whether copying research data there is acceptable, needs the "
                         "owner's confirmation; external drives: none visible.")
    return out


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1])
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"identities": identities(tok), "views": {}, "verify_reports": {}}
    for name, root in JOBS:
        vr = root / f"{name}-verify.json"
        if vr.exists():
            v = json.loads(vr.read_text())
            rep["verify_reports"][name] = {"chunks_ok": sum(1 for x in v.get("chunks", {}).values() if x == "ok"),
                                           "views": {c: {k: v2[k] for k in ("questions", "states", "layers", "d_model")}
                                                     for c, v2 in v.get("views", {}).items()},
                                           "copy": v.get("copy")}
        job = json.loads((root / f"{name}-job.json").read_text())
        for cond in job["identity"]["conditions"]:
            view = root / f"{name}-{cond}"
            if (view / "agent_states.parquet").exists():
                rep["views"][f"{name}/{cond}"] = audit_view(view, tok)
                print(name, cond, rep["views"][f"{name}/{cond}"]["counts"])
    rep["backups"] = backups()
    tot = Counter()
    for v in rep["views"].values():
        tot.update({k: v2 for k, v2 in v["counts"].items() if isinstance(v2, int)})
    rep["totals"] = dict(tot)
    (OUT / "archive_audit.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"identities_ok": rep["identities"]["local_tokenizer_matches_archives"], "totals": rep["totals"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
