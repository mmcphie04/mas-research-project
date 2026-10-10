import os, subprocess, sys
code, out = "/content/mas-research-project", "/content/mas_runs"
os.makedirs(out, exist_ok=True)
cmd = [sys.executable, "scripts/run_experiment.py", "--config", "configs/base.yaml", "configs/colab_12b.yaml",
       "--out-root", out, "--name", "pilot12b", "--dataset", "mmlu-pro", "--pool", "pilot",
       "--n-questions", "120", "--chunk", "20",
       "--conditions", "self_reconsider", "debate", "debate_answer_only", "misleading_one",
       "--smoke", "--max-hours", "1", "--set", "model.batch_size=8", "model.readout_batch_size=4"]
log = open(f"{out}/smoke.log", "w")
p = subprocess.Popen(cmd, cwd=code, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "PYTHONPATH": f"{code}/src", "PYTHONUNBUFFERED": "1"})
open(f"{out}/smoke.pid", "w").write(str(p.pid))
print("launched smoke, pid", p.pid)
