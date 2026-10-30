# Cross-model diversity run on main250: priced, NOT launched (exploratory)

**Status:** price and plan only (2026-10-30). Nothing here has run on a GPU. Everything this run
produces is **exploratory**: it is outside `docs/MAIN250_ANALYSIS_SPEC.md`.

**Question.** Gemma's 10 independent answers make errors that are strongly correlated:

- correctness ICC 0.70;
- effective number of voters about 1.3;
- two wrong answers agree 4.9 times as often as uniform spreading would give
  (`artifacts/main250/analysis/exploratory.md` §3).

Do answers from other model families make less correlated errors, and does a count-matched
mixed panel vote better than a Gemma-only one?

## Design

- **Questions:** the same 200 main250 questions. The selection is the same, and the job refuses
  to start unless the ids hash to `9a33c0ea…`.
- **Answers:** 5 independent first answers per question from each of two non-Gemma model
  families. Condition `pool` with `debate.n_agents=5`; no debate.
- **Protocol:** the same as Gemma's. 250-word guidance (`brevity_words: 250`), 1,280-token cap,
  same prompts and templates.
- **Option permutations:** the same. They are a function of question id and
  `data.permutation_seed` only, not of the model. After the run, every question's displayed
  answer key must equal main250's; that check makes answers comparable across models letter for
  letter.
- **Kept:** answers, stated-answer parses, readouts (letter distributions, fp32 head, batch 1),
  verbalized confidence, response log-prob summaries and token counts.
- **No activation storage.**
- One job per model, run one after the other in one session.

## Proposed checkpoints

| role | checkpoint | size | licence / access | CPU checks done 2026-10-30 (tokenizer and config files only) |
|---|---|---|---|---|
| family 1 | `Qwen/Qwen2.5-7B-Instruct` | 7.6 B | Apache-2.0, ungated | letters A–J are distinct single tokens and pass `_validate_letter_context` after `"\nFinal answer: ("`; system role supported; eos `<|im_end|>` |
| family 2 | `meta-llama/Llama-3.1-8B-Instruct` | 8.0 B | Llama 3.1 licence, gated; **your token has access** (its files downloaded) | letters pass; system role supported; no pad token (the backend already sets pad = eos); eos ids [128001, 128008, 128009] |
| fallback | `ibm-granite/granite-3.1-8b-instruct` | 8.2 B | Apache-2.0, ungated | letters pass; system role supported; pad = eos (id 0) |
| rejected | `mistralai/Mistral-7B-Instruct-v0.3` | 7.2 B | | **fails** the letter-context check: every letter is spelled differently after `(` (SentencePiece), so the readout would score the wrong tokens |

Revisions are pinned to the commit hash at launch and recorded in the job identity (as for Gemma).

## Backend changes needed (CPU work and tests before any GPU)

1. **Sampling parity (required).**
   - `HFBackend.generate` passes only `temperature` and `top_p`. Every other sampling setting
     comes from each model's own `generation_config.json`.
   - Gemma at G4's commit `96b6f1ec` has `top_k: 64` and no repetition penalty, so G4 sampled
     with temperature 0.7, top_p 0.95, top_k 64.
   - Unchanged, Qwen would sample with **top_k 20 and repetition_penalty 1.05**, and Llama with
     the library default **top_k 50**.
   - The change: add `ModelConfig.top_k` and `ModelConfig.repetition_penalty`, passed to
     `generate` explicitly. They are versioned like `gen_batch_tokens`: omitted from the
     identity when unset, so legacy hashes are unchanged.
   - Set `top_k: 64, repetition_penalty: 1.0` for both new models. That matches Gemma's
     effective settings.
2. **Chat templates.** No code change. Each model is rendered by its own tokenizer template
   (`apply_chat_template`), and the template hash is recorded.
   - **Qwen2.5:** our prompts always send a system message, so Qwen's built-in default system
     prompt is *not* inserted.
   - **Llama 3.1:** the template always adds "Cutting Knowledge Date: December 2023 / Today
     Date: 26 Jul 2024" to the system turn. It is the same fixed text every time; recorded,
     not changed.
   - **Both:** load through `AutoModelForCausalLM` (the existing non-Gemma path). Stop ids come
     from each generation config (existing).
3. **No activation storage.**
   - Add `activations.store: false` (versioned as in item 1).
   - The writer then writes no shards, and the completion records and views accept an
     activation-free directory.
   - Readout still computes the last hidden state, which the fp32 head needs.
4. **Smoke first.** On the GPU, a 2-question `--smoke` per model before its full job checks:
   - template rendering;
   - stop tokens;
   - stated-answer parse and the before-letter position;
   - readout replay;
   - memory.

## Estimated GPU time and units (estimates; not measured)

| item | basis | per model | both models |
|---|---|---|---|
| responses | 200 × 5 = 1,000 | | 2,000 |
| generation + readout | Gemma-12B round 0: **5.97 s/response** with readout (G4, measured). 7–8 B models decode about 1.5× faster (weights 15–16 GB vs 24 GB; smaller KV cache: Qwen 56 KB/token, Llama 128 KB vs Gemma 384 KB). Lengths under the same 250-word guidance are unknown → 0.5–0.85× Gemma → **3.0–5.1 s/response** | 0.83–1.4 h | 1.7–2.8 h |
| download + load + stress + smoke | Gemma: download 1.1 min, load 1.6 min, stress 6.6 min; pool-only stress is shorter (worst prompt ≈ 1.1 k tokens) | 0.15–0.25 h | 0.3–0.5 h |
| session setup, final backup and verification | outputs are small without activations (≈ tens of MB per model) | | 0.1–0.2 h |
| **total** | | | **2.1–3.5 h ≈ 14–24 units** at 6.73/h |

**Proposed limits:**

- hard cap **25 units**: stop line = balance at launch − 25;
- `--max-hours 1.8` per job as a backstop.

Supervision as for G4: per-chunk backup, a 6 GB memory cap, the runtime always stopped, and
the laptop kept awake.

## Planned measurements (all exploratory; fixed before the run)

Agent indices are prespecified: each model's agents 0..m−1. Gemma's are pool agents
0..m−1 from main250. Every comparison is paired by question, with 95% intervals from 10,000
question resamples.

1. **Per model:**
   - accuracy per agent, and plurality over its 5;
   - stated-answer rate and fallback rate;
   - share capped at 1,280 tokens;
   - length distribution.

   Each model is reported next to Gemma's numbers. Lower accuracy can outweigh lower
   correlation.
2. **Cross-model error correlation:** the correlation of correctness (phi) between two answers
   to the same question, for every model pair:
   - within Gemma, within Qwen and within Llama;
   - Gemma–Qwen, Gemma–Llama and Qwen–Llama.

   Also: the ICC within each model; the joint wrong rate against the product of the marginals;
   n_eff for mixed sets.
3. **Same-wrong-answer rate:** when both answers are wrong, how often they name the same wrong
   option, within vs across models, against the uniform expectation 1/(n_options − 1). This
   relies on the shared permutation.
4. **Mixed-panel plurality curves, count-matched**, scored as exact expected accuracy under
   random ties:

   | panel size | Gemma-only | mixed |
   |---|---|---|
   | 3 | G0–2 | 1G + 1Q + 1L |
   | 6 | G0–5 | 2G + 2Q + 2L; 3G + 3Q; 3G + 3L |
   | 9 | G0–8 | **3G + 3Q + 3L** |

   - Plus single-family curves: Q-only and L-only at 3 and 5.
   - **Headline exploratory contrast:** 3G + 3Q + 3L vs 9G, paired by question.
   - Robustness: the exact average over all agent subsets within each model, instead of the
     first m.
   - Coverage (at least one correct answer) and the coverage-minus-plurality gap for each panel.
5. **Power note:** main250's paired standard deviations (0.23–0.33) imply a minimum detectable
   effect of about 5–7 points at n = 200. A mixed-panel gain smaller than that will not be
   resolvable.
