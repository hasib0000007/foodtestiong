## 5. Part B: Sensor Telemetry Branch Training

A channel-aware MLP (64–32–16 with BatchNorm and dropout) trained on the
harmonized 16-D feature table to classify Fresh / Stale / Chemically-Masked
states. Training curves, the confusion matrix, per-class precision/recall/F1 and
one-vs-rest ROC curves are reported on the held-out validation split.
