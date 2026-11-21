"""Stage complete generation chunks (mad.gen.archive) onto a fresh runtime, ONE CHUNK PER UPLOAD,
each checked on the runtime against its own completion record before the next is sent. Chunks
already complete on the runtime are skipped (a rerun resumes).

    python scripts/stage_chunks.py --session l2 --src runs/expab --job expab \\
        --parts bank_qwen_nt__all_orig bank_qwen_t__all_orig --remote-root /content/expab

Also copies the job record. Never sends anything the backup has not verified locally first.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mad.job import _check_hashes  # noqa: E402
from mad.provenance import is_complete  # noqa: E402

MARK = "<<<STAGE_JSON>>>"


class ColabChunkStager:
    def __init__(self, session, code_dir, remote_root):
        from colab_backup import ColabRemote

        self.cli = ColabRemote(session, code_dir, remote_root, "/content/backup_state", "unused")
        self.code_dir, self.root = code_dir, remote_root

    def _run(self, body, timeout=900):
        code = (f"import json, os, sys, tarfile\nsys.path.insert(0, {self.code_dir + '/src'!r})\n"
                f"from mad.job import _check_hashes\nfrom mad.provenance import is_complete\n{body}\n")
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
        return self._run(f"print({MARK!r} + json.dumps([n for n in {list(names)!r} "
                         f"if is_complete(os.path.join({self.root!r}, n))]))", timeout=300)

    def install(self, tar: Path, name: str):
        self.cli._colab("upload", "-s", self.cli.session, str(tar), "/content/stage_chunk.tar", timeout=3600)
        body = (f"from pathlib import Path\nroot = Path({self.root!r}); root.mkdir(parents=True, exist_ok=True)\n"
                f"with tarfile.open('/content/stage_chunk.tar') as t: t.extractall(root, filter='data')\n"
                f"_check_hashes(root / {name!r})\nos.remove('/content/stage_chunk.tar')\n"
                f"print({MARK!r} + json.dumps('ok'))")
        return self._run(body)

    def put_file(self, local: Path, remote_name: str):
        self.cli._colab("upload", "-s", self.cli.session, str(local), f"{self.root}/{remote_name}", timeout=600)


def stage(stager, src: Path, job: str, parts: list[str], log=print) -> dict:
    dirs = sorted(d for p in parts for d in src.glob(f"{job}-{p}-c*") if d.is_dir() and ".tmp-" not in d.name)
    for d in dirs:
        if not is_complete(d):
            raise SystemExit(f"{d.name} is not complete locally")
        _check_hashes(d)  # only verified chunks are sent
    have = set(stager.present([d.name for d in dirs]))
    sent, t0 = [], time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for d in dirs:
            if d.name in have:
                continue
            tar = Path(tmp) / "c.tar"
            with tarfile.open(tar, "w") as t:
                t.add(d, arcname=d.name)
            stager.install(tar, d.name)
            sent.append(d.name)
            log(json.dumps({"staged": d.name, "mb": round(tar.stat().st_size / 1e6, 1)}))
            tar.unlink()
    job_file = src / f"{job}-job.json"
    if job_file.exists():
        stager.put_file(job_file, job_file.name)
    return {"chunks": len(dirs), "sent": len(sent), "skipped": len(dirs) - len(sent), "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--src", default=str(ROOT / "runs/expab"))
    ap.add_argument("--job", default="expab")
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--remote-root", default="/content/expab")
    ap.add_argument("--code-dir", default="/content/mas-research-project")
    a = ap.parse_args(argv)
    out = stage(ColabChunkStager(a.session, a.code_dir, a.remote_root), Path(a.src), a.job, a.parts,
                log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
