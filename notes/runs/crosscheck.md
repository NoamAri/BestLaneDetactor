# Tracker cross-check

Tuning window: `data/out/tracking/T8_D1_0-30s` (selected there: `r2_botsort_first005`)  
Held-out window: `data/out/tracking/T8_D1_30-30s` (selected there: `r2_botsort_first005`)

| Variant | Step | Fragmentation, tuning (vs baseline) | Jumps, tuning | Guards, tuning | Fragmentation, held-out (vs baseline) | Jumps, held-out | Guards, held-out |
|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 478 (+0%) | 19 | pass | 547 (+0%) | 21 | pass |
| `s1_lowconf` | 1 | 395 (-17%) | 28 | fail | 479 (-12%) | 26 | pass |
| `s2_bt_buf2s` | 2 | 382 (-20%) | 28 | fail | 472 (-14%) | 27 | pass |
| `s2_bt_buf4s` | 2 | 379 (-21%) | 29 | fail | 471 (-14%) | 27 | pass |
| `s2_bt_buf6s` | 2 | 379 (-21%) | 29 | fail | 471 (-14%) | 27 | pass |
| `s2_bt_buf4s_iou03` | 2 | 359 (-25%) | 99 | fail | 466 (-15%) | 64 | fail |
| `s2_bt_buf4s_act04` | 2 | 501 (+5%) | 52 | fail | 607 (+11%) | 51 | fail |
| `s2_ocsort_buf4s` | 2 | 259 (-46%) | 44 | fail | 324 (-41%) | 38 | fail |
| `s2_botsort_buf4s` | 2 | 462 (-3%) | 0 | pass | 527 (-4%) | 2 | pass |
| `r2_botsort_first01` | 2b | 408 (-15%) | 2 | pass | 484 (-12%) | 6 | pass |
| `r2_botsort_first005` **(final)** | 2b | 378 (-21%) | 6 | pass | 449 (-18%) | 10 | pass |
| `r2_botsort_first01_unc02` | 2b | 408 (-15%) | 2 | pass | 484 (-12%) | 6 | pass |
| `r2_ocsort_iou03` | 2b | 362 (-24%) | 23 | fail | 405 (-26%) | 27 | fail |
| `r2_ocsort_iou03_buf2s` | 2b | 416 (-13%) | 19 | fail | 473 (-14%) | 22 | fail |
| `r2_bt_conf015_buf4s` | 2b | 414 (-13%) | 28 | fail | 481 (-12%) | 15 | pass |

Final choice (selection rule on the held-out window): `r2_botsort_first005`
