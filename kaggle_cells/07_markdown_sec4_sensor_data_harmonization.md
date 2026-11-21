## 4. Sensor Data Harmonization & Augmentation

3–4 Kaggle sensor CSV datasets are loaded through per-dataset column mappings into
the 8 canonical channels. Missing channels are detected, flagged (availability
flags) and imputed from cross-dataset statistics; a coverage report enumerates
which extra datasets close the remaining gaps. The masked class (label 2) is trained
exclusively on real formalin-confirmed QC rows with inverse-frequency class weighting,
and storage-environment channels not covered by any attached dataset receive
documented synthetic variation. Min–max scaling and a moving-average pre-filter
complete the harmonization.
