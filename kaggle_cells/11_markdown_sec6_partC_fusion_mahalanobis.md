## 6. Part C: Multimodal Fusion & Mahalanobis Anomaly Detection

Both branches are frozen; only the compact fusion engine trains, taking the
concatenation [Fv 128 | Fs 16 | scaled sensor values 16 | state posterior 3] and
regressing the continuous Freshness Index (0–100). The Mahalanobis distance is
calibrated on the fresh-class fused embeddings with a shrunk covariance estimate;
samples that appear fresh but lie far outside the fresh-feature distribution
trigger the chemical-masking alert.
