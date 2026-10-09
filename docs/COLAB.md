# Running on Colab

The laptop runs the 1B model at roughly 220 seconds per question. Detecting the effects this
project is looking for needs roughly 600 to 4,000 questions, depending on the effect size and on
whether only changed answers count, so on the laptop it takes days. Colab Pro for Education
provides an A100 with 40 GB, which fits Gemma-3-12B, the model the plan was written for.

## Two ways to run

**Google's official Colab CLI (preferred).** `google-colab-cli` from the `googlecolab` organisation
drives a Colab runtime from the laptop: `colab new -s mas --gpu A100`, `colab upload`,
`colab exec -f script.py`, `colab download`, `colab usage`, `colab stop`. This is the sanctioned way
to control Colab from outside a browser; unofficial tunnels into a runtime are what Colab's terms
prohibit. Installed in WSL with `uv tool install google-colab-cli`, and it signs in through
Application Default Credentials (needs `gcloud`, installed user-locally at `~/google-cloud-sdk`).
Pass `--auth=adc` explicitly — the CLI's own help and its bundled docs disagree about the default.

Known limits, from the CLI's bundled agent guide (`colab skill`):
- `colab exec -f script.py` sends one file. The `mad` package must be uploaded as well.
- `colab drivemount` and `colab auth` need a person at the terminal, so unattended runs should
  download finished chunks to the laptop rather than depend on Drive.
- Whether a job keeps running after the local command disconnects is not documented. Long runs
  should launch as a background process on the runtime and be checked with short commands.

**The notebook (fallback).** Open `notebooks/colab_runner.ipynb` in Colab and run all cells; results
go to Google Drive under `MyDrive/mas_runs/`. Instructions follow.

## One-time setup

1. Open the notebook in Colab: *File → Open notebook → GitHub*, search `mmcphie04/mas-research-project`
   (authorise Colab to see private repositories), choose `notebooks/colab_runner.ipynb`.
2. *Runtime → Change runtime type → A100 GPU.*
3. Add two **Colab Secrets** (key icon, left sidebar) and turn on *Notebook access* for both:
   - `HF_TOKEN` — Hugging Face → Settings → Access Tokens → a **Read** token. Must belong to the
     account that accepted the Gemma licence.
   - `GITHUB_TOKEN` — GitHub → Settings → Developer settings → Fine-grained tokens. Repository
     access: *only* `mas-research-project`. Permissions: *Contents → Read-only*.
     (*Read and write* only if you want the optional final cell to push results to a branch.)

## Running

Edit the settings cell, then *Runtime → Run all*. The cells:

| cell | what it does |
|---|---|
| 2 | confirms a GPU is attached |
| 3 | mounts Drive |
| 4 | reads the two secrets; never prints them |
| 5 | clones or pulls the repo; masks the token in any git error; removes it from the saved remote |
| 6 | installs pinned dependencies, keeping Colab's CUDA build of torch |
| 7 | **smoke test**: 2 questions through every condition |
| 8 | full run, resumable |
| 9 | summary with 95% intervals |
| 10 | optional: push small results to the `colab-results` branch (off by default; never `master`) |

**Run the smoke test first and stop if it fails.** It asserts that every stored state has the shape
the analysis expects — 49 layers × 3,840 numbers for Gemma-3-12B — and reports seconds per question
and peak GPU memory, then projects the full run's duration.

## What has and has not been verified

Verified locally:
- 101 tests pass, including the driver, resume after a lost chunk, the smoke checks, the summary,
  and checks that the notebook parses, contains no secrets, masks the token and never pushes to
  `master`.
- The driver runs end to end on the stub exactly as the notebook calls it.
- The MMLU-Pro loader on the real dataset: 12,032 questions, 14 subjects, every gold answer valid.
- Gemma-3-12B's configuration, on an empty "meta" device with no weights downloaded: 48 layers,
  d = 3,840, 26.4 GB of weights in bf16, answer letters A–J each a single distinct token.

**Not verified until the first Colab run:**
- That hidden states actually come back from the 12B model. Its forward function does not take
  `output_hidden_states` as a named argument; it passes it through a catch-all keyword parameter.
  The 1B model uses the identical mechanism and returned all 27 layers, so it should work — the
  smoke test asserts it explicitly rather than assuming.
- Real throughput and memory. Estimates below.
- That the pinned `transformers==5.17.0` installs against Colab's torch.

The `vllm` backend is still not implemented. The Hugging Face backend runs on the GPU; it is slower
at generation than vLLM but returns hidden states directly, which vLLM does not.

## Memory

Weights take 26.4 GB of the 40. The remainder holds the KV cache during generation (~0.39 GB per
1,000 tokens in the batch) and hidden states during readout (~0.38 GB per 1,000 tokens). A debate
round-1 prompt contains the agent's own answer plus two peers' answers, so at 768 generation tokens
a sequence approaches ~3,400 tokens. Batch 4 keeps peak near 32 GB; batch 8 would approach 37.
`configs/colab_12b.yaml` defaults to 4. The driver logs the worst-case estimate at start-up, and the
smoke test reports the measured peak; raise `model.batch_size` in `OVERRIDES` only if that shows room.

## Budget

300 compute units at about 5.3 per hour ≈ 56 A100-hours.

Rough estimates, to be replaced by the smoke test's measurement: generation on a single A100 is
limited by reading 26 GB of weights per step, so roughly 20–40 seconds per question across three
conditions.

| run | questions | estimated time | estimated units |
|---|---|---|---|
| smoke test | 2 | 5–10 min, mostly the model download | ~1 |
| pilot | 60 | 30–50 min | 3–5 |
| powered run | 1,500 | 8–17 h | 45–90 |

## Getting results back to the laptop

Run directories go to Drive. The laptop has no Drive sync, so one of:

1. **Install Google Drive for Desktop** on Windows. Drive then appears as a drive letter that WSL can
   read directly, including the activation files. Simplest for analysis.
2. **Run the summary on Colab** (cell 9) and read `<OUT_ROOT>/<RUN_NAME>-summary.json`.
3. **Push small results** with the last cell, then `git fetch origin colab-results` on the laptop.
   Activation files are not pushed — they are too large for git.

## Token budget

On the 1B model, 192 tokens still cut off 73% of first answers. 768 is a guess for the 12B model on
MMLU-Pro; the smoke test and pilot report the cut-off rate. If it is high, raising the budget is the
wrong fix — it multiplies cost — and a brevity instruction in the prompt is the right one.
