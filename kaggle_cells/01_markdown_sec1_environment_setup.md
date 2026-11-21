## 1. Environment Setup & Configuration

Imports, random seeding, canonical channel definitions, Kaggle dataset discovery
helpers, and the per-dataset configuration for the attached image and sensor
datasets.

Each sensor entry maps its own column names onto the 8 canonical channels; a
mapping value of `None` means the dataset has no data for that channel — the
loader records an availability flag and imputes the value from cross-dataset
statistics. The chemically masked class (label 2) trains on REAL formalin-confirmed
QC rows from the Formalin Tofu Aroma dataset. `MASK_AUGMENT_FRACTION` can add a
physics-based simulated component, but it is disabled by default so every class is
trained on real data only; class weighting compensates for the remaining imbalance.
