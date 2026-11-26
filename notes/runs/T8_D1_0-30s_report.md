# Tracking experiment

Detections: `data/interim/T8_D1_0-30s_detections.parquet`  
Frames 0-899

| Variant | Step | IDs | Mid-image starts+ends | Starts | Ends | < 1 s | Restart pairs | Median life (s) | Identity jumps | Clear-box coverage | Guards |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 386 | 478 | 244 | 234 | 102 | 119 | 3.67 | 19 | 99.5% | pass |
| `s1_lowconf` | 1 | 349 | 395 | 206 | 189 | 74 | 98 | 5.37 | 28 | 99.5% | identity jumps 28 > limit 25 |
| `s2_bt_buf2s` | 2 | 343 | 382 | 200 | 182 | 71 | 90 | 5.71 | 28 | 99.5% | identity jumps 28 > limit 25 |
| `s2_bt_buf4s` | 2 | 341 | 379 | 199 | 180 | 70 | 91 | 5.71 | 29 | 99.5% | identity jumps 29 > limit 25 |
| `s2_bt_buf6s` | 2 | 341 | 379 | 199 | 180 | 70 | 91 | 5.71 | 29 | 99.5% | identity jumps 29 > limit 25 |
| `s2_bt_buf4s_iou03` | 2 | 331 | 359 | 190 | 169 | 59 | 86 | 6.24 | 99 | 99.5% | identity jumps 99 > limit 25 |
| `s2_bt_buf4s_act04` | 2 | 417 | 501 | 262 | 239 | 98 | 115 | 4.47 | 52 | 99.8% | identity jumps 52 > limit 25 |
| `s2_ocsort_buf4s` | 2 | 268 | 259 | 129 | 130 | 44 | 49 | 6.99 | 44 | 95.3% | identity jumps 44 > limit 25; coverage 0.954 < limit 0.985 |
| `s2_botsort_buf4s` | 2 | 376 | 462 | 236 | 226 | 92 | 131 | 4.05 | 0 | 99.3% | pass |
| `r2_botsort_first01` | 2b | 348 | 408 | 209 | 199 | 75 | 106 | 4.99 | 2 | 99.4% | pass |
| `r2_botsort_first005` **(selected)** | 2b | 333 | 378 | 194 | 184 | 64 | 95 | 5.47 | 6 | 99.4% | pass |
| `r2_botsort_first01_unc02` | 2b | 348 | 408 | 209 | 199 | 75 | 106 | 4.99 | 2 | 99.4% | pass |
| `r2_ocsort_iou03` | 2b | 324 | 362 | 181 | 181 | 76 | 80 | 4.49 | 23 | 95.3% | coverage 0.953 < limit 0.985 |
| `r2_ocsort_iou03_buf2s` | 2b | 352 | 416 | 209 | 207 | 89 | 104 | 3.75 | 19 | 95.3% | coverage 0.953 < limit 0.985 |
| `r2_bt_conf015_buf4s` | 2b | 358 | 414 | 216 | 198 | 81 | 104 | 4.45 | 28 | 99.5% | identity jumps 28 > limit 25 |

Selected: `r2_botsort_first005`
