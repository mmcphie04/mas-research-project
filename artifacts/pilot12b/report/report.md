# Pilot report: pilot12b

_pilot data choose instrument settings and scope; they do not test the activation hypothesis._

| condition | 1 agent | vote of 3 | after round 1 | panel optimal | panel optimal − better baseline | parsed (pre/post) | cap hit (pre/post) | B / H / U (distinct q) |
|---|---|---|---|---|---|---|---|---|
| self_reconsider | 0.561 [0.48, 0.64] | 0.575 [0.49, 0.66] | 0.483 [0.39, 0.57] | 0.675 [0.59, 0.76] | 0.100 [0.05, 0.16] | 0.950 / 0.992 | 0.050 / 0.008 | 35 (25) / 79 (44) / 67 (38) |
| debate | 0.561 [0.48, 0.64] | 0.575 [0.49, 0.66] | 0.558 [0.47, 0.65] | 0.650 [0.57, 0.73] | 0.075 [0.03, 0.11] | 0.950 / 1.000 | 0.050 / 0.000 | 32 (27) / 32 (23) / 46 (22) |
| debate_answer_only | 0.561 [0.48, 0.64] | 0.575 [0.49, 0.66] | 0.575 [0.48, 0.66] | 0.650 [0.57, 0.73] | 0.075 [0.03, 0.10] | 0.950 / 1.000 | 0.050 / 0.000 | 32 (26) / 26 (20) / 45 (23) |
| misleading_one | 0.561 [0.48, 0.64] | 0.575 [0.49, 0.66] | 0.550 [0.47, 0.63] | 0.650 [0.57, 0.73] | 0.075 [0.03, 0.12] | 0.950 / 1.000 | 0.050 / 0.000 | 30 (28) / 59 (45) / 75 (39) |

**Runtime:** 74.2 s/question measured over 120 completed questions ({'round0': 19.3, 'overhead': 0.1, 'self_reconsider': 17.6, 'debate': 11.8, 'debate_answer_only': 13.2, 'misleading_one': 12.3}); projected 48.7 s/question for self_reconsider, debate; GPU-hours {600: 8.1, 900: 12.2, 1200: 16.2}.

**Disk:** projected GB {600: 10.9, 900: 16.3, 1200: 21.8}.

**Power** (test questions needed, 80% power, two-sided 5%):

| discordant | effect | test q | total q |
|---|---|---|---|
| 0.05 | 2 pts | 933 | 1866 |
| 0.05 | 3 pts | 386 | 772 |
| 0.05 | 4 pts | 190 | 380 |
| 0.10 | 2 pts | 1939 | 3878 |
| 0.10 | 3 pts | 849 | 1698 |
| 0.10 | 4 pts | 467 | 934 |
| 0.15 | 2 pts | 2928 | 5856 |
| 0.15 | 3 pts | 1293 | 2586 |
| 0.15 | 4 pts | 720 | 1440 |
| 0.20 | 2 pts | 3913 | 7826 |
| 0.20 | 3 pts | 1733 | 3466 |
| 0.20 | 4 pts | 970 | 1940 |
