"""Stage complete generation chunks (mad.gen.archive) onto a fresh runtime in BATCHES of chunks
(one tar of up to ``--batch-mb`` per upload), every chunk checked on the runtime against its own
completion record. Chunks already complete on the runtime are skipped (a rerun resumes).

Each CLI call has a short timeout and a stalled batch is retried: the runtime is asked which of its
chunks are already installed, and only the rest are re-sent (install is idempotent). Seen live in
launch 4: ``colab exec`` calls stalled past their own timeout at random chunks while the runtime
stayed responsive, and per-chunk staging (~420 calls for 210 small chunks) hit one stall per
~10 chunks.

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

    def install(self, tar: Path, names: list[str], timeout=240):
        self.cli._colab("upload", "-s", self.cli.session, str(tar), "/content/stage_batch.tar", timeout=600)
        body = (f"from pathlib import Path\nimport hashlib\nroot = Path({self.root!r}); root.mkdir(parents=True, exist_ok=True)\n"
                f"tp = '/content/stage_batch.tar'\n"
                f"with tarfile.open(tp) as t: t.extractall(root, filter='data')\n"
                f"st = Path('/content/backup_state'); st.mkdir(parents=True, exist_ok=True)\n"
                f"p = st / 'acked.json'\nak = json.load(open(p)) if p.exists() else {{}}\n"
                f"for n in {list(names)!r}:\n"
                f"    _check_hashes(root / n)\n"
                # the runtime's backup state: a staged chunk is already on the laptop -- acknowledge it
                # (stamp = sha256 of its completion record, as mad.cli_backup) so it is not sent back
                f"    ak[n] = hashlib.sha256(open(root / n / 'completion.json', 'rb').read()).hexdigest()\n"
                f"json.dump(ak, open(p, 'w'))\nos.remove(tp)\n"
                f"print({MARK!r} + json.dumps('ok'))")
        return self._run(body, timeout=timeout)

    def put_file(self, local: Path, remote_name: str):
        self.cli._colab("upload", "-s", self.cli.session, str(local), f"{self.root}/{remote_name}", timeout=600)


def batches(dirs: list[Path], batch_mb: float) -> list[list[Path]]:
    out, cur, size = [], [], 0
    for d in dirs:
        sz = sum(f.stat().st_size for f in d.iterdir())
        if cur and size + sz > batch_mb * 1e6:
            out.append(cur)
            cur, size = [], 0
        cur.append(d)
        size += sz
    return out + ([cur] if cur else [])


def stage(stager, src: Path, job: str, parts: list[str], log=print, batch_mb: float = 40, tries: int = 4) -> dict:
    dirs = sorted(d for p in parts for d in src.glob(f"{job}-{p}-c*") if d.is_dir() and ".tmp-" not in d.name)
    for d in dirs:
        if not is_complete(d):
            raise SystemExit(f"{d.name} is not complete locally")
        _check_hashes(d)  # only verified chunks are sent
    have = set(stager.present([d.name for d in dirs]))
    sent, t0 = [], time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for batch in batches([d for d in dirs if d.name not in have], batch_mb):
            todo = batch
            for attempt in range(1, tries + 1):
                tar = Path(tmp) / "b.tar"
                with tarfile.open(tar, "w") as t:
                    for d in todo:
                        t.add(d, arcname=d.name)
                mb = round(tar.stat().st_size / 1e6, 1)
                try:
                    stager.install(tar, [d.name for d in todo])
                    sent += [d.name for d in todo]
                    log(json.dumps({"staged": len(todo), "first": todo[0].name, "last": todo[-1].name, "mb": mb, "attempt": attempt}))
                    break
                except RuntimeError as exc:  # a stalled or failed call: re-send only what is not installed
                    log(json.dumps({"retry": attempt, "chunks": len(todo), "error": str(exc)[:200]}))
                    if attempt == tries:
                        raise
                    for _ in range(3):
                        try:
                            got = set(stager.present([d.name for d in todo]))
                            break
                        except RuntimeError:
                            time.sleep(10)
                    else:
                        got = set()
                    sent += [d.name for d in todo if d.name in got]
                    todo = [d for d in todo if d.name not in got]
                    if not todo:
                        break
                finally:
                    tar.unlink(missing_ok=True)
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
    ap.add_argument("--batch-mb", type=float, default=40)
    a = ap.parse_args(argv)
    out = stage(ColabChunkStager(a.session, a.code_dir, a.remote_root), Path(a.src), a.job, a.parts,
                log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True), batch_mb=a.batch_mb)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
