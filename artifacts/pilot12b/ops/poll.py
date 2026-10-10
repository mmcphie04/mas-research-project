import re, subprocess
out = "/content/mas_runs"
name = open("/content/poll_target").read().strip()
pid = int(open(f"{out}/{name}.pid").read())
alive = subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0
lines = [l for l in open(f"{out}/{name}.log", errors="replace").read().splitlines()
         if "Loading weights" not in l and "it/s]" not in l and "examples/s]" not in l]
pat = re.compile(r"model loaded|stress|c\d{3}:|FAILED|stopping|out of memory|Traceback|Error|time cap|finished|incomplete|VIEW|SMOKE")
print(f"[{name}] process {'RUNNING' if alive else 'EXITED'}; milestones:")
print("\n".join(l[:400] for l in lines if pat.search(l)))
print("--- last lines:")
print("\n".join(l[:300] for l in lines[-4:]))
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
