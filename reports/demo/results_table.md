| Language | Type | Model | Test accuracy | Macro-F1 | Model latency (ms/sample) | Split | n_test |
|---|---|---|---|---|---|---|---|
| PSL | static | RF | 0.961 | 0.966 | 8.02 | cross_signer (SYNTHETIC) | 640 |
| PSL | static | MLP | 0.958 | 0.963 | 2.10 | cross_signer (SYNTHETIC) | 640 |
| PSL | dynamic | LSTM | 0.990 | 0.990 | 38.64 | cross_signer (SYNTHETIC) | 300 |
| PSL | dynamic | GRU | 1.000 | 1.000 | 55.25 | cross_signer (SYNTHETIC) | 300 |
| ASL | static | RF | 1.000 | 1.000 | 7.77 | cross_signer (SYNTHETIC) | 720 |
| ASL | static | MLP | 1.000 | 1.000 | 1.60 | cross_signer (SYNTHETIC) | 720 |
| ASL | dynamic | LSTM | 0.997 | 0.997 | 36.17 | cross_signer (SYNTHETIC) | 300 |
| ASL | dynamic | GRU | 1.000 | 1.000 | 59.06 | cross_signer (SYNTHETIC) | 300 |

| Language | Type | Pipeline mean (ms/frame) | p95 (ms) | FPS (pipeline only) |
|---|---|---|---|---|
| PSL | static | 2.38 | 2.59 | 421 |
| PSL | dynamic | 15.89 | 52.94 | 63 |
| ASL | static | 2.68 | 2.91 | 373 |
| ASL | dynamic | 15.34 | 49.44 | 65 |
