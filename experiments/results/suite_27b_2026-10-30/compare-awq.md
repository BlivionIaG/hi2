## bench vs bench

| task | n | ref % | cand % | diff [95% CI] | McNemar p | R->W % | W->R % | recovery | n for 1 pt |
|---|---|---|---|---|---|---|---|---|---|
| ifeval | 100 | 95.0 | 94.0 | -1.0 [-6.0, +4.0] | 1 | 4.0 | 3.0 | 0.989 | 5492 |
| needle | 68 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 1 | 0.0 | 0.0 | 1.000 | - |
| mc | 100 | 64.0 | 68.0 | +4.0 [+1.0, +8.0] | 0.125 | 0.0 | 4.0 | 1.062 | 3138 |
| decision | 66 | 87.9 | 81.8 | -6.1 [-12.1, -1.5] | 0.125 | 6.1 | 0.0 | 0.931 | 4755 |
| gsm8k | 100 | 95.0 | 95.0 | +0.0 [+0.0, +0.0] | 1 | 0.0 | 0.0 | 1.000 | - |
| humaneval | 100 | 94.0 | 96.0 | +2.0 [+0.0, +5.0] | 0.5 | 0.0 | 2.0 | 1.021 | 1568 |
| mbpp | 100 | 93.0 | 89.0 | -4.0 [-9.0, +0.0] | 0.219 | 5.0 | 1.0 | 0.957 | 4707 |
| toolcall | 97 | 60.8 | 49.5 | -11.3 [-19.6, -3.1] | 0.0127 | 14.4 | 3.1 | 0.814 | 13754 |
| **all (pooled)** | 731 | 86.2 | 84.2 | -1.9 [-3.6, -0.3] | 0.0385 | | | 0.972 | |

ref/cand % in the last row are macro means over tasks; diff and p are pooled over all items. R->W: reference right, candidate wrong.
n for 1 pt: paired items for 80% power to detect a 1-point difference (alpha 0.05, McNemar) at this pair's discordance rate (- when no item flipped); about 78500 x the discordance.

| task | measure | ref | cand |
|---|---|---|---|
| ifeval | mean frac | 0.9650 | 0.9600 |
| mc | mean acc_norm | 0.6500 | 0.6400 |
| decision | agree on tool | 100 | 84.8 [74.3, 91.6] |
| decision | agree on tool+arg | 100 | 72.7 [61.0, 82.0] |
| toolcall | mean arg_f1 | 0.7175 | 0.6181 |
