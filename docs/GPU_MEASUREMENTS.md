# GPU-dependent measurements: what is saved, what is not

Everything below needs the model on a GPU and cannot be recomputed after the runtime is gone
without paying for inference again. "Verified" means checked on CPU with the 270M Gemma 3 (same
tokenizer family), not on the 12B. **No 12B measurement has been made yet.**

## Saved for every agent state

| measurement | where | phases | CPU-verified |
|---|---|---|---|
| full option-letter distribution at the forced `Final answer: (` | `readout_logprobs` | pre, exposure, post | yes |
| hidden states, all 49 layers, at `readout_last` (forced `(`) | activation shards | pre, exposure, post | yes |
| hidden states at `generation_last` (last response token; last prompt token for exposure) | shards | pre, exposure, post | yes |
| hidden states at `stated_answer` (the letter in the agent's own final answer line) | shards | pre, post when the answer parses | yes |
| hidden states at `before_stated_answer` (the token whose output predicts that letter) | shards | pre, post when the answer parses | yes |
| option-letter distribution at each extra position (letter rows of the output layer, with the model's soft-capping) | `pos_<name>_logprobs` | as above | yes — equals the model's own logits at the readout position |
| token index and token id of each extra position; reason for each missing position | `pos_<name>_token_index/_id/_missing` | as above | yes |
| exact-replay flag: the readout input is the generation-time token sequence | `replay_exact` | pre, post | yes (the readout is built from generated ids, not re-tokenised text) |
| response log-likelihood at temperature 1 (mean, min, sum of token log-probabilities) | `response_logprob_*` | pre, post | yes — equals full-sequence logits |
| generated token ids, stop reason (stop token or cap), prompt and response token counts | transcripts, rows | pre, post | yes |
| structured messages and the rendered prompt | transcripts | pre, post | yes |
| stated answer, parse status (strict / permissive / missing / out of range), verbalized confidence | rows | pre, post | yes |
| what each agent was shown: peer order, sources, shown answers (missing kept missing) | rows, transcripts | post | yes |
| seeds per chunk and condition; batch sizes per session | manifests, job history | — | yes |

## Saved per job and session

| measurement | where | CPU-verified |
|---|---|---|
| model commit, tokenizer, chat template hash, special tokens, letter token ids, dtype | job identity | yes |
| GPU name, memory, CUDA, NVIDIA driver, package versions | job history (each session) | partly (no GPU on CPU) |
| token-length coverage of the whole intended question pool | `<name>-stress.json`, log | yes |
| worst-case memory at each candidate batch size (generation and readout), finite values at all layers | `<name>-stress.json` | CPU only (no CUDA peaks) |
| batched vs one-at-a-time readout agreement | smoke report | yes (exact in fp32); **bf16 not verified** |
| stage timings (round 0, generation, readouts, writes) per chunk; session wall time | status file, ledger | yes |

## Optional collections (separate commands)

| measurement | command | CPU-verified |
|---|---|---|
| candidate evaluation: correct/incorrect candidate × neutral/endorse-correct/endorse-incorrect, distributions + states | `scripts/run_candidates.py` | stub only |
| three fresh independent answers per question | child job, condition `independent` | stub only |
| answer-only peers on a matched subset | child job, condition `debate_answer_only` | stub only |

## Deliberately not saved

| measurement | why not |
|---|---|
| full-vocabulary logits at any position | 262,144 × 4 bytes per position; letter distributions and response likelihoods carry what the analyses use |
| hidden states at every token position | ~1,000× the storage of the chosen positions |
| attention weights, KV caches | storage and memory; no planned analysis uses them |
| per-token log-probabilities of the response | only mean, min and sum are kept; per-token values would need ~3 KB per response — add `response_logprob_tokens` if wanted before the main run |
| generation-time hidden states | states come from a replay of the exact generated token sequence (`replay_exact`), which is identical for a causal model up to numerical batching effects |
| bit-identical resampling under a different batch size | Hugging Face has no per-sequence generator; batch sizes are logged per session |

## Must be checked on the first 12B run (the smoke test does each)

1. Hidden states come back for all 49 layers at every requested position, finite and not
   degenerate.
2. The letter-in-context check passes on the 12B tokenizer (it runs at model load).
3. Batched and one-at-a-time readouts agree in bf16 within tolerance.
4. Every generated row replays exactly (`replay_exact` true).
5. Memory at the worst-case lengths of the actual pool, at batch 4, 6 and 8.
6. Real seconds per question, per stage.
