# Tracking experiment

Detections: `data/interim/T8_D1_0-30s_detections.parquet`  
Frames 0-899  
Guards relative to: `r2_botsort_first005`

| Variant | Step | IDs | Mid-image starts+ends | Starts | Ends | < 1 s | Restart pairs | Median life (s) | Identity jumps | Clear-box coverage | Stitches | Guards |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 386 | 478 | 244 | 234 | 102 | 119 | 3.67 | 19 | 99.5% | 0 | identity jumps 19 > limit 9 |
| `r2_botsort_first005` **(selected)** | 2b | 333 | 378 | 194 | 184 | 64 | 95 | 5.47 | 6 | 99.4% | 0 | pass |
| `s3_stitch_2s` | 3 | 289 | 290 | 150 | 140 | 52 | 53 | 6.47 | 36 | 99.4% | 44 | identity jumps 36 > limit 9 |
| `s3_stitch_5s` | 3 | 253 | 218 | 114 | 104 | 40 | 21 | 7.14 | 45 | 99.4% | 80 | identity jumps 45 > limit 9 |
| `s3_stitch_5s_loose` | 3 | 222 | 156 | 83 | 73 | 29 | 1 | 10.53 | 80 | 99.4% | 111 | identity jumps 80 > limit 9 |

Selected: `r2_botsort_first005`
