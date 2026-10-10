import sys, os, time
sys.path.insert(0, "/content/mas-research-project/src")
os.chdir("/content/mas-research-project")
import numpy as np, torch
from mad.config import load_config
from mad.backends.base import build_backend
from mad import prompts as P
from mad.permute import identity_permutation
cfg = load_config("configs/base.yaml", "configs/colab_12b.yaml",
                  **{"activations.positions": ["readout_last"]})
t = time.time(); b = build_backend(cfg); print(f"loaded in {time.time()-t:.0f}s")
msgs = P.round0_messages(persona=cfg.debate.personas[0], subject="physics",
    question="Which quantity is conserved in an elastic collision?",
    options=["mass only", "momentum only", "kinetic energy and momentum", "neither"],
    perm=identity_permutation(4), ask_confidence=True)
text = b.render(msgs) + "Both are conserved.\nAnswer: (C)\nConfidence: 70"
ids = torch.tensor([b.tokenizer.encode(text + P.READOUT_SUFFIX, add_special_tokens=False)], device="cuda")
with torch.no_grad():
    res = b.model(input_ids=ids, output_hidden_states=True, logits_to_keep=1, use_cache=False)
print("dtype of hidden states:", res.hidden_states[0].dtype)
print("layer  max|x| (all positions)  max|x| (last position)  nonfinite_raw  >65504(last pos)")
worst = 0
for L, h in enumerate(res.hidden_states):
    hf = h[0].float()
    last = hf[-1]
    big = int((last.abs() > 65504).sum())
    worst = max(worst, float(last.abs().max()))
    if L % 4 == 0 or big or not torch.isfinite(hf).all():
        print(f"{L:5d}  {float(hf.abs().max()):14.1f}  {float(last.abs().max()):14.1f}  "
              f"{int((~torch.isfinite(hf)).sum()):6d}  {big:6d}")
print("max |x| at the readout position over all layers:", worst)
x16 = torch.stack([h[0, -1] for h in res.hidden_states]).float().cpu().numpy().astype(np.float16)
print("after float16 cast: non-finite values =", int((~np.isfinite(x16.astype(np.float32))).sum()),
      "in layers", sorted(set(np.where(~np.isfinite(x16.astype(np.float32)))[0].tolist())))
