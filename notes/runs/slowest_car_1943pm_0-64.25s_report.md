# Tracking experiment

Detections: `data/interim/slowest_car_1943pm_0-64.25s_detections.parquet`  
Frames 0-256  
Guards relative to: `baseline`

| Variant | Step | IDs | Mid-image starts+ends | Starts | Ends | < 1 s | Restart pairs | Median life (s) | Identity jumps (in pieces) | Flags at stitches | Clear-box coverage | Stitches | Guards |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 156 | 171 | 136 | 35 | 11 | 3 | 6.5 | 83 | 0 | 93.0% | 0 | pass |
| `s1_lowconf` | 1 | 153 | 159 | 133 | 26 | 11 | 1 | 6.5 | 83 | 0 | 93.1% | 0 | pass |
| `s2_bt_buf2s` | 2 | 149 | 151 | 129 | 22 | 10 | 0 | 6.5 | 85 | 0 | 93.2% | 0 | pass |
| `s2_bt_buf4s` | 2 | 149 | 151 | 129 | 22 | 10 | 0 | 6.5 | 85 | 0 | 93.2% | 0 | pass |
| `s2_bt_buf6s` | 2 | 149 | 150 | 129 | 21 | 10 | 0 | 6.5 | 85 | 0 | 93.2% | 0 | pass |
| `s2_bt_buf4s_iou03` | 2 | 160 | 154 | 136 | 18 | 12 | 0 | 6.5 | 104 | 0 | 95.9% | 0 | identity jumps 104 > limit 102 |
| `s2_bt_buf4s_act04` | 2 | 151 | 154 | 130 | 24 | 10 | 1 | 6.75 | 86 | 0 | 94.1% | 0 | pass |
| `s2_ocsort_buf4s` | 2 | 162 | 230 | 142 | 88 | 37 | 4 | 5.5 | 133 | 0 | 76.9% | 0 | identity jumps 133 > limit 102; coverage 0.769 < limit 0.920 |
| `s2_botsort_buf4s` | 2 | 63 | 65 | 41 | 24 | 4 | 2 | 12.75 | 4 | 0 | 56.0% | 0 | coverage 0.560 < limit 0.920 |
| `r2_botsort_first01` | 2b | 61 | 59 | 39 | 20 | 3 | 0 | 11.25 | 29 | 0 | 57.2% | 0 | coverage 0.572 < limit 0.920 |
| `r2_botsort_first005` | 2b | 65 | 57 | 42 | 15 | 6 | 0 | 8.25 | 43 | 0 | 59.1% | 0 | coverage 0.591 < limit 0.920 |
| `r2_botsort_first01_unc02` | 2b | 79 | 78 | 54 | 24 | 7 | 0 | 7.25 | 17 | 0 | 63.0% | 0 | coverage 0.630 < limit 0.920 |
| `r2_ocsort_iou03` | 2b | 63 | 82 | 53 | 29 | 10 | 1 | 8.25 | 9 | 0 | 54.2% | 0 | coverage 0.542 < limit 0.920 |
| `r2_ocsort_iou03_buf2s` | 2b | 67 | 88 | 57 | 31 | 13 | 3 | 7.5 | 6 | 0 | 54.6% | 0 | coverage 0.546 < limit 0.920 |
| `r2_bt_conf015_buf4s` **(selected)** | 2b | 147 | 148 | 127 | 21 | 8 | 0 | 6.5 | 86 | 0 | 93.2% | 0 | pass |

Selected: `r2_bt_conf015_buf4s`
