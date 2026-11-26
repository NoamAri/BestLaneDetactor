# Tracking experiment

Detections: `data/interim/T8_D1_30-30s_detections.parquet`  
Frames 899-1797

| Variant | Step | IDs | Mid-image starts+ends | Starts | Ends | < 1 s | Restart pairs | Median life (s) | Identity jumps | Clear-box coverage | Guards |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 427 | 547 | 281 | 266 | 116 | 150 | 3.27 | 21 | 99.4% | pass |
| `s1_lowconf` | 1 | 396 | 479 | 250 | 229 | 101 | 111 | 3.44 | 26 | 99.4% | pass |
| `s2_bt_buf2s` | 2 | 393 | 472 | 247 | 225 | 98 | 107 | 3.6 | 27 | 99.4% | pass |
| `s2_bt_buf4s` | 2 | 393 | 471 | 247 | 224 | 98 | 107 | 3.77 | 27 | 99.4% | pass |
| `s2_bt_buf6s` | 2 | 393 | 471 | 247 | 224 | 98 | 107 | 3.8 | 27 | 99.4% | pass |
| `s2_bt_buf4s_iou03` | 2 | 389 | 466 | 245 | 221 | 90 | 106 | 3.87 | 64 | 99.4% | identity jumps 64 > limit 27 |
| `s2_bt_buf4s_act04` | 2 | 485 | 607 | 314 | 293 | 146 | 153 | 2.9 | 51 | 99.7% | identity jumps 51 > limit 27 |
| `s2_ocsort_buf4s` | 2 | 296 | 324 | 163 | 161 | 59 | 52 | 5.87 | 38 | 95.0% | identity jumps 38 > limit 27; coverage 0.950 < limit 0.984 |
| `s2_botsort_buf4s` | 2 | 412 | 527 | 272 | 255 | 122 | 141 | 2.67 | 2 | 99.2% | pass |
| `r2_botsort_first01` | 2b | 390 | 484 | 251 | 233 | 108 | 125 | 3.39 | 6 | 99.2% | pass |
| `r2_botsort_first005` **(selected)** | 2b | 372 | 449 | 233 | 216 | 98 | 109 | 3.92 | 10 | 99.3% | pass |
| `r2_botsort_first01_unc02` | 2b | 390 | 484 | 251 | 233 | 108 | 125 | 3.39 | 6 | 99.2% | pass |
| `r2_ocsort_iou03` | 2b | 344 | 405 | 206 | 199 | 102 | 84 | 4.59 | 27 | 94.9% | coverage 0.949 < limit 0.984 |
| `r2_ocsort_iou03_buf2s` | 2b | 379 | 473 | 239 | 234 | 125 | 108 | 3.07 | 22 | 94.9% | coverage 0.949 < limit 0.984 |
| `r2_bt_conf015_buf4s` | 2b | 396 | 481 | 251 | 230 | 96 | 114 | 3.8 | 15 | 99.4% | pass |

Selected: `r2_botsort_first005`
