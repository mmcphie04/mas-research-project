# Runtime side of the laptop backup: pack every complete directory (completion.json present) not
# yet packed, plus the job's small files (job, questions, status, ledger, stress, smoke reports,
# logs), into /content/sync/sync.tar. Directories are only ever complete after an atomic rename,
# so a packed directory is never half-written.
import json, os, tarfile
out = "/content/mas_runs"
prefix = open("/content/sync_target").read().strip()
os.makedirs("/content/sync", exist_ok=True)
state = "/content/sync/packed.json"
packed = json.load(open(state)) if os.path.exists(state) else {}
items, sizes = [], 0
for p in sorted(os.listdir(out)):
    if not p.startswith(prefix) or ".tmp-" in p or ".verify-" in p:
        continue
    full = os.path.join(out, p)
    if os.path.isdir(full):
        comp = os.path.join(full, "completion.json")
        if not os.path.exists(comp):
            continue
        stamp = os.path.getmtime(comp)
        if packed.get(p) == stamp:
            continue
        packed[p] = stamp
    items.append(p)
    sizes += sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(full) for f in fs) \
        if os.path.isdir(full) else os.path.getsize(full)
with tarfile.open("/content/sync/sync.tar", "w") as t:
    for p in items:
        t.add(os.path.join(out, p), arcname=p)
json.dump(packed, open(state, "w"))
print(f"packed {len(items)} items, {sizes / 1e6:.1f} MB: {[i for i in items if os.path.isdir(os.path.join(out, i))]}")
