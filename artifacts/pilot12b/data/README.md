# Pilot review export: gemma-3-12b-it, 120 MMLU-Pro pilot-pool questions

A **review export**, not a self-contained activation archive: every table and record of the
pilot except the activation arrays. Directory and file names are the archive's own.

- **Collection commit:** `5ff52b38e332546106c18463317f07a21b8b0ae1` (both collection sessions).
- **Exported from:** the laptop archive `runs/colab/` (3.6 GB), after `verify_archive.py` passed
  on the complete archive (`pilot12b-verify.json`): all checksums of 24 chunk directories, views
  rebuilt with checksum verification enabled and compared, all 49 layers × 3,840 finite.
- **Activations:** stored as **bfloat16** bit patterns (uint16) in `acts_*.npy` and
  `round0_hidden_*.npy`; decode with `mad.store.decode` (or `ActivationStore`), never
  `.astype(float)`. **Omitted here** (3.5 GB): all activation arrays. The indexes
  (`activation_index.parquet`) and shapes (`activation_meta.parquet`) are included.
- **Lineage:** chunks c000–c002 (questions 1–60) were collected in session 2; c003–c005
  (questions 61–120) in session 3, after the first attempt at them was lost unseen with a
  reclaimed runtime (docs/PILOT_LOG.md). Same commit, settings, prompts and derived seeds.

## Contents

| path | what |
|---|---|
| `pilot12b-questions.jsonl` | every pilot question: id, subject, text, options, gold index (original order) |
| `pilot12b-job.json` | job identity (config, prompt hashes, model/tokenizer identity), ordered question hashes, chunks, history (`created`, `resumed`) |
| `pilot12b-status.json`, `pilot12b-ledger.jsonl` | progress and budget records; the ledger is **incomplete** (session 2's copy after the last backup was lost) |
| `pilot12b-stress.json` | the pilot job's stress report (generation batch 8, readout batch 4) |
| `pilot12b-verify.json` | final independent verification of the complete laptop archive |
| `pilot12b-<condition>/` | merged view per condition: `agent_states.parquet` (one row per question × agent × phase), full `transcripts.jsonl`, `manifest.json`, `view.json`, `activation_index.parquet`, `activation_meta.parquet` |
| `pilot12b-<condition>-cNNN/` | each chunk's `manifest.json` and `completion.json` (file checksums, stage timings) |
| `pilot12b-round0-cNNN/` | each chunk's shared round 0: `round0.jsonl` (messages, rendered prompt, text, generated token ids, stop reason, letter distributions, alignment), `manifest.json`, `completion.json` |
| `logs/` | session-3 pilot log, the session-3 watcher log, and the milestones the session-2 monitor saw |

Conditions: `self_reconsider`, `debate`, `debate_answer_only`, `misleading_one`.

## Reading notes

- **Question ids are strings** (digit strings such as `"3248"`). With pandas read JSONL as
  `pd.read_json(path, lines=True, dtype={"question_id": str})`; the default turns them into
  integers and joins with the parquet tables then silently match nothing.
- `agent_states.parquet` holds, per state: canonical `answer` and `answer_source`
  (stated / readout_fallback / readout), `correct`, `readout_answer` and the full
  `readout_logprobs`, option distributions at each extra position (`pos_<position>_logprobs`,
  with `_token_index`, `_token_id`, `_missing` reason), `stated_status`, `stop_reason`,
  `verbalized_confidence`, response likelihood, `gold_display` and the option `permutation`
  (display order → original index), what the agent was shown (`shown_peer_answers`,
  `shown_peer_sources` in display order, `misleading_target`) and what its panel answered
  (`panel_peer_answers`), `request_id`, `gen_seed`.
- Phases: `pre` (round 0), `peer_exposure` (readout after reading the round-1 message, before
  writing), `post` (revised answer).
