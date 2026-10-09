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

Set `COMMIT` in the settings cell to the full hash of the commit to run (the notebook refuses
anything else and never follows a moving branch), edit the other settings, then *Runtime → Run
all*. The cells:

| cell | what it does |
|---|---|
| 2 | settings, including the pinned `COMMIT`, the pool (`pilot`/`main`) and `MAX_HOURS` |
| 4 | confirms a GPU is attached |
| 6 | mounts Drive |
| 8 | reads the two secrets; never prints them |
| 10 | checks out exactly `COMMIT`; masks the token in any git error; removes it from the saved remote |
| 12 | installs pinned dependencies, keeping Colab's CUDA build of torch |
| 15 | **smoke test** in a throwaway job: worst-case memory stress, then 2 questions through every condition |
| 17 | the run, resumable |
| 19 | summary with 95% intervals and paired differences |
| 21 | optional: push small results to a new `colab-results/<name>-<time>` branch (off by default; never `master`, never force) |

**Run the smoke test first and stop if it fails.** It forces the longest generation and readout the
protocol can produce and reports allocated and reserved memory peaks. It then checks every stored
state: 49 layers × 3,840 numbers for Gemma-3-12B, finite, not all identical, letter distributions
that sum to one, answers inside each question's own option range. It also checks that a batched
readout agrees with a one-at-a-time readout. Finally it reports seconds per question and projects
the full run.

### What makes a resume safe

The first session writes `<name>-job.json`: the dataset and its fixed question order, chunk
size, conditions, every setting except batch sizes, prompt hashes, the model's commit, tokenizer,
chat template and letter token ids, and the code commit. Every later session must match it or
the driver stops with a message saying what differs. Asking for more questions appends to the
same order (`<name>-questions.jsonl` keeps a copy of every question's text). Round 0 for each
chunk is saved before any condition runs and reloaded on retry. Each condition directory is
written under a temporary name and renamed only when complete. After the chunks, each condition
gets a merged *view* (`<name>-<condition>/`) that checks checksums, unique keys and exact question
membership and points at the chunk files instead of copying them.

The driver stops on an out-of-memory error, after two consecutive failed chunks, or once the
job's total wall time across sessions reaches `--max-hours`. `<name>-status.json` records state,
progress, per-chunk timings and memory peaks, and every failure.

### Commands (what the notebook runs)

```bash
CFG="--config configs/base.yaml configs/colab_12b.yaml --out-root /content/drive/MyDrive/mas_runs"
# 1. smoke + stress (throwaway job "smoke-pilot12b")
python scripts/run_experiment.py $CFG --name pilot12b --dataset mmlu-pro --pool pilot \
    --n-questions 120 --chunk 20 --conditions self_reconsider debate misleading_one --smoke
# 2. pilot: the 120-question pilot pool
python scripts/run_experiment.py $CFG --name pilot12b --dataset mmlu-pro --pool pilot \
    --n-questions 120 --chunk 20 --conditions self_reconsider debate misleading_one --max-hours 6
python scripts/summarize_run.py --out-root /content/drive/MyDrive/mas_runs --name pilot12b
# 3. main run, only after docs/preregistration.md is dated: N from the pilot, main pool
python scripts/run_experiment.py $CFG --name main12b --dataset mmlu-pro --pool main \
    --n-questions N --chunk 25 --conditions self_reconsider debate misleading_one --max-hours H
```

Extending a job later: rerun with a larger `--n-questions`, keeping it a multiple of `--chunk`
from a full last chunk (the driver refuses an extension that would change a chunk that already
has output).

## What has and has not been verified

Verified locally, on CPU:
- The full test suite, including the job-file refusals (changed settings, chunk size, pool,
  conditions, commit; a chunk from another job; a corrupted shard), extension by appending,
  round-0 reuse after a crash between conditions, the time cap persisting across sessions, the
  fail-fast stops, the merged view loading the same arrays as the chunks, and the notebook's
  pinning and push rules.
- The driver end to end on the stub, and on the real Hugging Face path with Gemma-3-270M on CPU
  (`--smoke`): the stress pass, the extra activation positions, generated token ids and stop
  reasons, and batched-vs-single readout agreement (identical in fp32 on CPU).
- The letter-in-context check (the tokenizer encodes `Final answer: (` + letter as the scored
  letter token) on the Gemma 3 tokenizer of the 270M model; it runs again at load time on the 12B.
- The MMLU-Pro loader on the real dataset: 12,032 questions, 14 subjects, every gold answer valid.
- Gemma-3-12B's configuration, on an empty "meta" device with no weights downloaded: 48 layers,
  d = 3,840, 26.4 GB of weights in bf16, answer letters A–J each a single distinct token.

**Not verified until the first Colab run (none of these has been run):**
- That hidden states come back from the 12B model, and are finite and non-degenerate.
- Real throughput and memory peaks under the stress pass, in bf16 on the A100.
- Batched-vs-single readout agreement in bf16 (it is exact in fp32 on CPU; bf16 with left
  padding can differ slightly, and the smoke test fails only on gross disagreement).
- That the pinned `transformers==5.17.0` installs against Colab's torch.

The `vllm` backend is still not implemented. The Hugging Face backend runs on the GPU; it is slower
at generation than vLLM but returns hidden states directly, which vLLM does not.

## Memory

Weights take 26.4 GB of the 40. The longest sequence the protocol produces is the round-1
readout: the question (up to ~600 tokens), the agent's own first answer, two peer answers and its
new answer (each up to the 768-token cap), plus ~250 tokens of instructions — about 3,900 tokens.
At that length the KV cache during generation is ~0.39 GB per 1,000 tokens per sequence and the
hidden states during readout ~0.38 GB per 1,000 tokens per sequence, so batch 4 needs ~6 GB for
each (an upper bound for the cache: most Gemma 3 layers use a 1,024-token sliding window), for a
peak around 33 GB of the 40. Generation and readout have separate batch sizes (`model.batch_size`,
`model.readout_batch_size`); both default to 4. The smoke test's stress pass measures the real
peaks; raise a batch size in `OVERRIDES` only if it shows room. Batch sizes may change between
sessions of a job; nothing else may.

## Budget

300 compute units at about 5.3 per hour ≈ 56 A100-hours.

Rough estimates, to be replaced by the smoke test's measurement: generation on a single A100 is
limited by reading 26 GB of weights per step, so roughly 20–40 seconds per question across three
conditions.

| run | questions | estimated time | estimated units |
|---|---|---|---|
| smoke test | 2 | 5–10 min, mostly the model download | ~1 |
| pilot | 120 | 1–1.5 h | 5–8 |
| powered run | 1,500 | 8–17 h | 45–90 |

## Getting results back to the laptop

Run directories go to Drive. The laptop has no Drive sync, so one of:

1. **Install Google Drive for Desktop** on Windows. Drive then appears as a drive letter that WSL can
   read directly, including the activation files. Simplest for analysis.
2. **Run the summary on Colab** (cell 9) and read `<OUT_ROOT>/<RUN_NAME>-summary.json`.
3. **Push small results** with the last cell, then on the laptop
   `git fetch origin 'refs/heads/colab-results/*:refs/remotes/origin/colab-results/*'`.
   Activation files are not pushed — they are too large for git.

## Token budget

On the 1B model, 192 tokens still cut off 73% of first answers. The prompts now ask for about 150
words of reasoning, and 768 tokens is a cap rather than a target. Every generated response
records why it stopped (`stop_reason`: a stop token or the cap), and the summary reports the
cap-hit rate. If it is high, tighten the brevity instruction — raising the cap multiplies cost.
