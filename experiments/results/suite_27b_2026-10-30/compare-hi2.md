## bench vs bench

| task | n | ref % | cand % | diff [95% CI] | McNemar p | R->W % | W->R % | recovery | n for 1 pt |
|---|---|---|---|---|---|---|---|---|---|
| ifeval | 100 | 95.0 | 95.0 | +0.0 [-6.0, +6.0] | 1 | 5.0 | 5.0 | 1.000 | 7847 |
| needle | 68 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 1 | 0.0 | 0.0 | 1.000 | - |
| mc | 100 | 64.0 | 61.0 | -3.0 [-9.0, +3.0] | 0.508 | 6.0 | 3.0 | 0.953 | 7062 |
| decision | 66 | 87.9 | 74.2 | -13.6 [-22.7, -6.1] | 0.00391 | 13.6 | 0.0 | 0.845 | 10701 |
| gsm8k | 100 | 95.0 | 96.0 | +1.0 [-2.0, +5.0] | 1 | 1.0 | 2.0 | 1.011 | 2353 |
| humaneval | 100 | 94.0 | 94.0 | +0.0 [-5.0, +5.0] | 1 | 3.0 | 3.0 | 1.000 | 4707 |
| mbpp | 100 | 93.0 | 84.0 | -9.0 [-16.0, -3.0] | 0.0117 | 10.0 | 1.0 | 0.903 | 8632 |
| toolcall | 97 | 60.8 | 55.7 | -5.2 [-14.4, +4.1] | 0.383 | 13.4 | 8.2 | 0.915 | 16991 |
| **all (pooled)** | 731 | 86.2 | 82.5 | -3.4 [-5.6, -1.2] | 0.00354 | | | 0.953 | |

ref/cand % in the last row are macro means over tasks; diff and p are pooled over all items. R->W: reference right, candidate wrong.
n for 1 pt: paired items for 80% power to detect a 1-point difference (alpha 0.05, McNemar) at this pair's discordance rate (- when no item flipped); about 78500 x the discordance.

| task | measure | ref | cand |
|---|---|---|---|
| ifeval | mean frac | 0.9650 | 0.9750 |
| mc | mean acc_norm | 0.6500 | 0.6600 |
| decision | agree on tool | 100 | 53.0 [41.2, 64.6] |
| decision | agree on tool+arg | 100 | 45.5 [34.0, 57.4] |
| toolcall | mean arg_f1 | 0.7175 | 0.7022 |
