"""Before collection: the validation questions (master positions 121-320) against the ACTUAL
pilot question ids, not only against job files present in an output root.

    uv run python artifacts/val12b/plan/check_disjoint.py
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
RE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RE)
from mad.config import load_config  # noqa: E402
from mad.job import master_order, select  # noqa: E402

cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/colab_12b.yaml"))
pool = RE.load_pool("mmlu-pro", cfg)
master = master_order(pool, cfg.data.sample_seed)
sha = hashlib.sha256(",".join(q.question_id for q in master).encode()).hexdigest()
pilot_job = json.loads((ROOT / "runs/colab/pilot12b-job.json").read_text())
pilot_ids = [q["id"] for q in pilot_job["questions"]]
assert sha == pilot_job["identity"]["pool"]["master_order_sha256"], "master order differs from the pilot's"
assert pilot_ids == [q.question_id for q in master[:120]], "pilot ids are not master positions 1-120"
val = select(master, "validation", 200, 120, 200)
val_ids = [q.question_id for q in val]
assert not set(val_ids) & set(pilot_ids), "validation overlaps the pilot"
main_first = [q.question_id for q in select(master, "main", 10, 120, 200)]
assert not set(main_first) & (set(val_ids) | set(pilot_ids))
subjects = {}
for q in val:
    subjects[q.subject] = subjects.get(q.subject, 0) + 1
rec = {"master_order_sha256": sha, "dataset_revision": cfg.data.dataset_revision, "order_seed": cfg.data.sample_seed,
       "pilot_ids_checked": len(pilot_ids), "validation_ids": val_ids,
       "validation_ids_sha256": hashlib.sha256(",".join(val_ids).encode()).hexdigest(),
       "subjects": dict(sorted(subjects.items())), "first_main_ids": main_first}
(Path(__file__).parent / "validation_question_ids.json").write_text(json.dumps(rec, indent=1))
print(f"master order matches the pilot's ({sha[:12]}); 200 validation ids disjoint from the 120 pilot ids;",
      f"subjects {len(subjects)} ({min(subjects.values())}-{max(subjects.values())} each)")
