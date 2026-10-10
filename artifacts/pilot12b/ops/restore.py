import os, tarfile, json
out = "/content/mas_runs"
os.makedirs(out, exist_ok=True)
with tarfile.open("/content/resume.tar") as t:
    t.extractall(out)
st = json.load(open(f"{out}/pilot12b-status.json"))
print("restored:", sorted(p for p in os.listdir(out))[:20])
print("status: chunks_done", st["chunks_done"], "elapsed_hours_total", st["elapsed_hours_total"])
