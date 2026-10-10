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

Run from a checkout of the pinned commit. Nothing below has been run on a GPU yet.

```bash
R=/content/drive/MyDrive/mas_runs
CFG="--config configs/base.yaml configs/colab_12b.yaml --out-root $R"
PILOT="--name pilot12b --dataset mmlu-pro --pool pilot --n-questions 120 --chunk 20 \
       --conditions self_reconsider debate debate_answer_only misleading_one"

# 1. smoke: worst-case memory stress at batches 4/6/8 over the pilot pool's real token lengths,
#    then 2 questions through all four conditions with value checks (throwaway job smoke-pilot12b)
python scripts/run_experiment.py $CFG $PILOT --smoke
#    read $R/smoke-pilot12b-stress.json and -smoke-report.json; raise model.batch_size with
#    --set model.batch_size=8 only if the stress report shows it fits

# 2. pilot: the 120-question pilot pool, all four conditions, capped at 4 hours
python scripts/run_experiment.py $CFG $PILOT --max-hours 4
python scripts/verify_archive.py --out-root $R --name pilot12b
python scripts/summarize_run.py  --out-root $R --name pilot12b
python scripts/pilot_report.py   --out-root $R --name pilot12b --main-n 600 900 1200 \
       --main-conditions 2 --out $R/pilot12b-report

# 3. main run -- only after docs/preregistration.md is dated. N and conditions from the pilot.
python scripts/run_experiment.py $CFG --name main12b --dataset mmlu-pro --pool main \
       --n-questions 900 --chunk 25 --conditions self_reconsider debate --max-hours 12
python scripts/make_split.py --out-root $R --name main12b --seed SEED --out $R/main12b-split.json

# 4. optional, reusing main12b's saved round 0 (never regenerated): answer-only peers and fresh
#    independent answers on the first 450 / 300 main questions; candidate evaluation on 300
python scripts/run_experiment.py $CFG --name main12b-answeronly --parent main12b --subset-n 450 \
       --conditions debate_answer_only --max-hours 3
python scripts/run_experiment.py $CFG --name main12b-indep --parent main12b --subset-n 300 \
       --conditions independent --max-hours 2
python scripts/run_candidates.py --config configs/base.yaml configs/colab_12b.yaml \
       --out-root $R --parent main12b --n 300 --split-manifest $R/main12b-split.json --splits train validation test

# 5. archive check before deleting anything from the runtime or moving it
python scripts/verify_archive.py --out-root $R --name main12b --copy-to /path/to/second/copy
```

A child job (`--parent`) uses the parent's dataset, pool, config, prompts and model, refuses any
difference, and reads round 0 from the parent's verified caches. Extending a job: rerun with a
larger `--n-questions`, a multiple of `--chunk` from a full last chunk.

## What has and has not been verified

Verified locally on CPU (stub and Gemma-3-270M; see docs/REVIEW_b56c836_CHANGES.md for the list):
the whole test suite; the driver end to end on the stub, including child jobs, the archive
verifier and its restored copy; and on the real Hugging Face path with the 270M model, the
stress pass over a 120-question pool, exact replay of every generated row, all four activation
positions with their option probabilities, response log-likelihoods equal to full-sequence
logits, and batched-vs-single agreement (exact in fp32).

**Not verified until the first 12B run** (each is checked by the smoke test; see
docs/GPU_MEASUREMENTS.md): hidden states for all 49 layers at every position, finite; the
letter-in-context check on the 12B tokenizer; bf16 batched-vs-single agreement; exact replay;
memory at the pool's worst-case lengths at batches 4/6/8; real seconds per question; that
`transformers==5.17.0` installs against Colab's torch.

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
peaks at batches 4, 6 and 8, using the longest prompt in the intended question pool (measured with
the real tokenizer) plus three full responses, and saves them to `<name>-stress.json`; raise a
batch size in `OVERRIDES` only if it shows room. A job's own stress pass runs once per GPU type
and is not repeated on resume. Batch sizes may change between
sessions of a job; nothing else may.

## Budget

300 compute units at about 5.3 per hour ≈ 56 A100-hours; the target is 10–20. See
**docs/BUDGET.md** for the per-question cost model, three plans (8–24 A100-hours including smoke,
pilot, extras and 15% contingency), power, and disk (~30 GB for the fullest plan: confirm the
Drive quota first). Every estimate is replaced by the pilot report's measurements.

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
