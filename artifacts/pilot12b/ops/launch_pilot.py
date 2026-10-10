import os, subprocess, sys
code, out = "/content/mas-research-project", "/content/mas_runs"
cmd = [sys.executable, "scripts/run_experiment.py", "--config", "configs/base.yaml", "configs/colab_12b.yaml",
       "--out-root", out, "--name", "pilot12b", "--dataset", "mmlu-pro", "--pool", "pilot",
       "--n-questions", "120", "--chunk", "20",
       "--conditions", "self_reconsider", "debate", "debate_answer_only", "misleading_one",
       "--max-hours", "3.5", "--skip-hash-check", "--set", "model.batch_size=8", "model.readout_batch_size=4"]
log = open(f"{out}/pilot.log", "a")
p = subprocess.Popen(cmd, cwd=code, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "PYTHONPATH": f"{code}/src", "PYTHONUNBUFFERED": "1"})
open(f"{out}/pilot.pid", "w").write(str(p.pid))
print("launched pilot, pid", p.pid)
