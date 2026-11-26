# Tracking experiment

Detections: `data/interim/T8_D1_30-30s_detections.parquet`  
Frames 899-1797  
Guards relative to: `r2_botsort_first005`

| Variant | Step | IDs | Mid-image starts+ends | Starts | Ends | < 1 s | Restart pairs | Median life (s) | Identity jumps (in pieces) | Flags at stitches | Clear-box coverage | Stitches | Guards |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0 | 427 | 547 | 281 | 266 | 116 | 150 | 3.27 | 21 | 0 | 99.4% | 0 | identity jumps 21 > limit 14 |
| `r2_botsort_first005` | 2b | 372 | 449 | 233 | 216 | 98 | 109 | 3.92 | 10 | 0 | 99.3% | 0 | pass |
| `s3_stitch_2s` | 3 | 315 | 335 | 176 | 159 | 72 | 54 | 5.34 | 10 | 35 | 99.3% | 57 | pass |
| `s3_stitch_5s` | 3 | 288 | 281 | 149 | 132 | 54 | 28 | 6.72 | 10 | 44 | 99.3% | 84 | pass |
| `s3_stitch_5s_loose` **(selected)** | 3 | 248 | 201 | 109 | 92 | 39 | 5 | 8.51 | 10 | 83 | 99.3% | 124 | pass |

Selected: `s3_stitch_5s_loose`
