# GPU-dependent measurements: what is saved, what is not

Everything below needs the model on a GPU and cannot be recomputed after the runtime is gone
without paying for inference again. "Verified" means checked on CPU with the 270M Gemma 3 (same
tokenizer family), not on the 12B. **No 12B measurement has been made yet.**

**Storage format: bfloat16** (2 bytes, the model's own compute format on the GPU, so lossless
there). The first A100 smoke attempt (2026-10-10) found that float16 cannot hold gemma-3-12b's
residual stream: one dimension reaches |x| = 163,840 at the readout position in layers 40-47
(over 800,000 at other positions, e.g. the first token), beyond float16's 65,504, so those values
were stored as infinity. The model's own bfloat16 values were all finite. Stored activations are
now bfloat16 bit patterns; readers decode to float32; older float16 runs still load.

## Saved for every agent state

| measurement | where | phases | CPU-verified |
|---|---|---|---|
| full option-letter distribution at the forced `Final answer: (` | `readout_logprobs` | pre, exposure, post | yes |
| hidden states, all 49 layers, at `readout_last` (forced `(`) | activation shards | pre, exposure, post | yes |
| hidden states at `generation_last` (last response token; last prompt token for exposure) | shards | pre, exposure, post | yes |
| hidden states at `stated_answer` (the letter in the agent's own final answer line) | shards | pre, post when the answer parses | yes |
| hidden states at `before_stated_answer` (the token whose output predicts that letter) | shards | pre, post when the answer parses | yes |
| option-letter distribution at each extra position (letter rows of the output layer, with the model's soft-capping) | `pos_<name>_logprobs` | as above | yes — equals the model's own logits at the readout position |
| token index and token id of each extra position; reason for each missing position (incl. `letter_token_mismatch`: the stated letter is not the standalone token the readout scores, so neither stated-answer state is stored) | `pos_<name>_token_index/_id/_missing` | as above | yes; a permissive `Answer: C` is flagged on the Gemma 3 tokenizer |
| readout input token counts (prompt part, own-response part) | `readout_prompt_tokens`, `readout_response_tokens` | all | tested |
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
| worst-case memory of the REAL workload at the configured batch sizes (plus optional extras): `generate` at full length, and `readout()` with all layers, all positions, option probabilities and the response likelihood; GPU allocated/reserved and host resident peaks; finite values; with the code, package, GPU and length context that decides whether it may be reused | `<name>-stress.json` | 270M on CPU, tested |
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

1. The model is 48 decoder layers × 3,840 (49 stored layers); every stored layer is finite and
   not degenerate, at every requested position.
2. The letter-in-context check passes on the 12B tokenizer (it runs at model load).
3. A fixed own-response transcript, with the agent's earlier answer and two peers' answers as
   distracting answer lines in the prompt: all four positions found on the right token ids; the
   stated letter is the scored letter token; the preceding token predicts it; option
   probabilities at every position match the model's own full logits within 0.02 (bf16); the
   prompt-only (exposure) readout ends at the prompt's last token and has no stated-answer
   states; a permissive `Answer: C` is either stored on the scored letter token or flagged.
4. Batched and one-at-a-time readouts, both fed the generated token ids, agree in bf16.
5. Every generated row replays exactly (`replay_exact` true); every record finite and normalised
   (checked before any chunk is marked complete).
6. Memory of the full extraction workload at the pool's worst-case lengths, at the configured
   batch sizes.
7. Real seconds per question, per stage.
