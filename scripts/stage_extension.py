"""Stage the parent's outputs a child job needs on a Colab runtime: the job files, then ONE CHUNK
AT A TIME (round 0 in full + the previous round's records without activation arrays; see
mad.stage), each hash-checked on the runtime before the next is uploaded.

    python scripts/stage_extension.py --session ext --parent main250 --src runs/main250 \\
        --subset artifacts/main250/plan/extension_subset.json \\
        --after debate3 structured3 debate5 structured5 --remote-root /content/main

Only chunks holding a subset question are staged. Chunks already installed on the runtime are
skipped, so a rerun after a failure resumes. Writes a JSON line per step to --log.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mad import stage as S  # noqa: E402
from mad.job import chunk_tag  # noqa: E402

MARK = "<<<STAGE_JSON>>>"


def needed_chunks(job: dict, subset: list[str]) -> list[int]:
    want = set(subset)
    idx = [i for i, members in enumerate(job["chunks"]) if want & set(members)]
    found = {q for i in idx for q in job["chunks"][i]} & want
    if found != want:
        raise SystemExit(f"{len(want - found)} subset questions are not in the parent job")
    return idx


class ColabStager:
    """Runtime side over the Colab CLI; the stage module is loaded by path from the checkout."""

    def __init__(self, session, code_dir, remote_root):
        from colab_backup import ColabRemote

        self.cli = ColabRemote(session, code_dir, remote_root, "/content/stage_state", "unused")
        self.code_dir, self.root = code_dir, remote_root

    def _run(self, body, timeout=900):
        code = (f"import json, importlib.util\n_s = importlib.util.spec_from_file_location('mad_stage_live', "
                f"{self.code_dir + '/src/mad/stage.py'!r})\nS = importlib.util.module_from_spec(_s)\n"
                f"_s.loader.exec_module(S)\n{body}\n")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
        try:
            out = self.cli._colab("exec", "-s", self.cli.session, "-f", fh.name, timeout=timeout)
        finally:
            Path(fh.name).unlink(missing_ok=True)
        if MARK not in out:
            raise RuntimeError(f"no result from the runtime: {out[-400:]}")
        return json.loads(out.split(MARK, 1)[1].strip().splitlines()[0])

    def present(self, names):
        return self._run(f"import os\nprint({MARK!r} + json.dumps([n for n in {list(names)!r} "
                         f"if os.path.exists(os.path.join({self.root!r}, n, 'completion.json')) "
                         f"or os.path.isfile(os.path.join({self.root!r}, n))]))", timeout=300)

    def install(self, tar: Path, manifest: dict):
        self.cli._colab("upload", "-s", self.cli.session, str(tar), "/content/stage_upload.tar", timeout=3600)
        m = {**manifest, "tar": "/content/stage_upload.tar"}
        return self._run(f"m = json.loads({json.dumps(m)!r})\nprint({MARK!r} + json.dumps(S.unpack_verified("
                         f"'/content/stage_upload.tar', m, {self.root!r}, '/content/stage_staging')))")


def stage(stager, src: Path, parent: str, subset: list[str], after: list[str], work: Path, log=print,
          attempts: int = 3) -> dict:
    job = json.loads((src / f"{parent}-job.json").read_text())
    idx = needed_chunks(job, subset)
    work.mkdir(parents=True, exist_ok=True)
    steps = [("job", None)] + [("chunk", i) for i in idx]
    done, total_bytes, t0 = [], 0, time.time()
    for kind, i in steps:
        tar = work / "stage.tar"
        m = (S.pack_job_files(src, parent, tar) if kind == "job"
             else S.pack_chunk(src, parent, chunk_tag(i), after, tar))
        names = sorted({r.split("/", 1)[0] for r in m["files"]})
        if set(stager.present(names)) == set(names):
            log(json.dumps({"step": kind, "chunk": i, "skipped": "already installed"}))
            continue
        for a in range(1, attempts + 1):
            try:
                t = time.time()
                got = stager.install(tar, m)
                total_bytes += m["bytes"]
                done.append(kind if i is None else chunk_tag(i))
                log(json.dumps({"step": kind, "chunk": i, "installed": len(got), "mb": round(m["bytes"] / 1e6, 1),
                                "seconds": round(time.time() - t, 1)}))
                break
            except Exception as exc:  # noqa: BLE001 -- retried; nothing is installed on a failed check
                log(json.dumps({"step": kind, "chunk": i, "attempt": a, "error": str(exc)[:300]}))
                if a == attempts:
                    raise
        tar.unlink(missing_ok=True)
    return {"staged": done, "chunks_needed": len(idx), "gb": round(total_bytes / 1e9, 2),
            "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--parent", default="main250")
    ap.add_argument("--src", default=str(ROOT / "runs/main250"))
    ap.add_argument("--subset", default=str(ROOT / "artifacts/main250/plan/extension_subset.json"))
    ap.add_argument("--after", nargs="+", default=["debate3", "structured3", "debate5", "structured5"])
    ap.add_argument("--remote-root", default="/content/main")
    ap.add_argument("--code-dir", default="/content/mas-research-project")
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    subset = json.loads(Path(a.subset).read_text())["question_ids"]
    work = Path(a.work or Path(a.src) / ".stage-work")
    out = stage(ColabStager(a.session, a.code_dir, a.remote_root), Path(a.src), a.parent, subset, a.after, work,
                log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
