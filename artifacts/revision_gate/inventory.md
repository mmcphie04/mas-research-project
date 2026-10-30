# Revision events in every Gemma-3-12B dataset (EXPLORATORY)

Helpful = wrong → right; harmful = right → wrong; wrong → wrong = changed between wrong answers. main250 test questions: counts only (reserved).

| dataset | condition | protocol (words) | panel | part | questions | changed | helpful | harmful | wrong→wrong | questions with ≥1 helpful/harmful | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot12b | debate | 150 | 3 | all | 120 | 110 | 32 | 32 | 46 | 31 |  |
| pilot12b | debate_answer_only | 150 | 3 | all | 120 | 103 | 32 | 26 | 45 | 33 | peers' answers only |
| pilot12b | misleading_one | 150 | 3 | all | 120 | 164 | 30 | 59 | 75 | 56 | one fabricated peer |
| pilot12b | self_reconsider | 150 | 3 | all | 120 | 181 | 35 | 79 | 67 | 60 | no peers |
| val12b | debate | 150 | 3 | all | 200 | 169 | 53 | 46 | 70 | 48 |  |
| val12b | independent | 150 | 3 | all | 200 | 160 | 51 | 46 | 63 | 64 | independent re-answer, not a revision |
| val12b | self_reconsider | 150 | 3 | all | 100 | 145 | 34 | 61 | 50 | 47 | no peers; 100 questions |
| pilot250 | debate3 | 250 | 3 | all | 12 | 13 | 6 | 4 | 3 | 4 |  |
| pilot250 | structured3 | 250 | 3 | all | 12 | 11 | 5 | 3 | 3 | 4 |  |
| pilot250 | debate5 | 250 | 5 | all | 12 | 23 | 7 | 8 | 8 | 4 |  |
| pilot250 | structured5 | 250 | 5 | all | 12 | 14 | 4 | 4 | 6 | 4 |  |
| pilot250_r2 | debate3_r2 | 250 | 3 | all | 4 | 2 | 1 | 1 | 0 | 1 | second round |
| pilot250_r2 | structured3_r2 | 250 | 3 | all | 4 | 2 | 2 | 0 | 0 | 1 | second round |
| pilot250_r2 | debate5_r2 | 250 | 5 | all | 4 | 5 | 2 | 3 | 0 | 1 | second round |
| pilot250_r2 | structured5_r2 | 250 | 5 | all | 4 | 0 | 0 | 0 | 0 | 0 | second round |
| pilot250_r2 | self_structured5 | 250 | 5 | all | 4 | 6 | 1 | 2 | 3 | 2 | no peers |
| main250 | debate3 | 250 | 3 | train+validation | 102 | 79 | 26 | 23 | 30 | 25 |  |
| main250 | debate3 | 250 | 3 | test (counts only) | 98 | 79 | 25 | 19 | 35 | 22 |  |
| main250 | structured3 | 250 | 3 | train+validation | 102 | 71 | 19 | 21 | 31 | 26 |  |
| main250 | structured3 | 250 | 3 | test (counts only) | 98 | 68 | 21 | 16 | 31 | 25 |  |
| main250 | debate5 | 250 | 5 | train+validation | 102 | 128 | 43 | 36 | 49 | 28 |  |
| main250 | debate5 | 250 | 5 | test (counts only) | 98 | 144 | 41 | 43 | 60 | 27 |  |
| main250 | structured5 | 250 | 5 | train+validation | 102 | 83 | 27 | 25 | 31 | 27 |  |
| main250 | structured5 | 250 | 5 | test (counts only) | 98 | 123 | 32 | 34 | 57 | 32 |  |

**Available for training (main250 test excluded):**

- all conditions: 410 helpful + 479 harmful = 889 events (580 wrong→wrong);
- revisions after seeing peers (the gate's sources): 284 helpful + 287 harmful = 571 events (397 wrong→wrong), 422 questions.

The gate uses only revisions made after seeing peers (its features describe the peers); self-reconsideration, independent re-answers and pilot250's 4-question second round are listed but not used. pilot250's 12 questions are validation-pool questions also in val12b; resampling and folds group by question id across datasets.
