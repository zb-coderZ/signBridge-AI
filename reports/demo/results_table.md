| Language | Type | Model | Test accuracy | Macro-F1 | Model latency (ms/sample) | Split | n_test |
|---|---|---|---|---|---|---|---|
| PSL | static | RF | 0.961 | 0.966 | 89.44 | cross_signer (SYNTHETIC) | 640 |
| PSL | static | MLP | 0.958 | 0.963 | 12.09 | cross_signer (SYNTHETIC) | 640 |
| PSL | dynamic | LSTM | 0.990 | 0.990 | 255.30 | cross_signer (SYNTHETIC) | 300 |
| PSL | dynamic | GRU | 1.000 | 1.000 | 382.85 | cross_signer (SYNTHETIC) | 300 |
| ASL | static | RF | 1.000 | 1.000 | 114.01 | cross_signer (SYNTHETIC) | 720 |
| ASL | static | MLP | 1.000 | 1.000 | 11.90 | cross_signer (SYNTHETIC) | 720 |
| ASL | dynamic | LSTM | 0.997 | 0.997 | 298.72 | cross_signer (SYNTHETIC) | 300 |
| ASL | dynamic | GRU | 1.000 | 1.000 | 442.88 | cross_signer (SYNTHETIC) | 300 |

| Language | Type | Pipeline mean (ms/frame) | p95 (ms) | FPS (pipeline only) |
|---|---|---|---|---|
| PSL | static | 9.47 | 16.27 | 106 |
| PSL | dynamic | 87.50 | 398.60 | 11 |
| ASL | static | 12.75 | 16.48 | 78 |
| ASL | dynamic | 100.80 | 370.93 | 10 |
