# Perfect Dataset Combo — Tri-Layer Multimodal Food Freshness Model

Curated for the thesis model: **separately-trained image branch (MobileNetV2 → Fv)**,
**separately-trained sensor branch (MLP → Fs)**, **feature-level late fusion +
Mahalanobis anti-masking layer**. All links verified; every choice maps to a
target group in the proposal (fish / meat / fruits).

## The Combo at a Glance

| Branch | Dataset | Role |
|---|---|---|
| A — Image (fruit) | [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) | Primary fruit/vegetable training set |
| A — Image (fish) | [Tilapia Fresh and Non Fresh (6,900 imgs)](https://www.kaggle.com/datasets/haripriyasanga/tilapia-fish-fresh-and-non-fresh-species) | Fish group (South-Asian species, day-wise decay) |
| A — Image (fish, extra) | [Fish Freshness Detection](https://www.kaggle.com/datasets/smailakgl/fish-freshness-detection) | Optional: anchovy / horse mackerel fresh-stale pairs |
| A — Image (meat) | [Meat Freshness Image Dataset (2,266 imgs)](https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset) | Meat group (Fresh / Half-Fresh / Spoiled) |
| A — Image (meat, cited) | [Meat Quality Assessment (ASYU 2019)](https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning) | Optional benchmark with published baseline accuracy |
| B — Sensor (real e-nose) | [Fruit Freshness Electronic Nose Dataset](https://www.kaggle.com/datasets/mehrabmahdian/food-freshness-electronic-nose-data) | Real MQ2–MQ135 telemetry, day-wise decay CSVs, citable DOI |
| B — Sensor (masking class) | Built-in synthetic generator (in the notebook) | Only source of **chemically-masked** samples (label 2) for the Mahalanobis layer |
| B — Sensor (validation) | [e-Nose Beef Quality (Mendeley)](https://data.mendeley.com/datasets/n8mc3nspfn/1) | 8 MOS + DHT11 logged on ESP32 → cross-checks the real-hardware claim |
| Context | [Food Adulteration Dataset (Bangladesh)](https://www.kaggle.com/datasets/fahmidachowdhury/food-adulteration-dataset) | Introduction/motivation evidence — not model input |

## Why this exact combination

1. **Covers all three proposal target groups** — fish (Tilapia), meat (beef/mutton),
   fruits (6 produce types) → matches the 200 fish + 200 meat + 200 fruit plan.
2. **Image folders are auto-discoverable** — every image dataset uses folder names
   containing `fresh` / `stale` / `rotten` / `spoiled`, so
   `notebooks/food_freshness_kaggle_training.ipynb` picks them up with
   **zero code change**.
3. **The e-nose dataset is real hardware telemetry** (MQ2–MQ135, 24 h sampling
   over multi-day decay, CC BY 4.0, DOI: 10.5281/zenodo.17285312) — this keeps the
   sensor branch on physically measured data instead of pure simulation, and gives a
   citable academic reference for the Data chapter.
4. **The masked class only exists in the synthetic generator.** No public dataset
   pairs formalin-treatment labels with gas readings — the notebook's 96-hour
   formalin-masking simulation fills that gap honestly (state this explicitly in
   the thesis: "augmented with controlled simulated adulteration").
5. **Big-data fallback:** [Food Freshness Dataset (70k images)](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset)
   for an ablation run showing accuracy vs. dataset size.

## Kaggle assembly steps

1. Attach image datasets: 3 inputs (fruit, fish, meat). Set
   `IMAGE_DATA_DIR` per run (one dataset path per experiment — run each branch
   training three times, then merge metrics for the thesis table).
2. Attach `food-freshness-electronic-nose-data`. Map its day-series CSVs to the
   notebook's 8 sensor columns, set `SENSOR_CSV` to the merged CSV (~38 MB total).
3. Keep `SENSOR_CSV = None` on the first run to validate the full pipeline with the
   synthetic generator before switching to real telemetry.
4. Compare: SYNTHETIC vs REAL-e-nose rows side by side in Results Table 6.2 —
   a strong defense point.

## Suggested citation line (Data chapter)

"Visual training data: Kaggle fresh/stale corpora [fruit, fish, meat]. Sensor
training data: a public electronic-nose fruit-freshness dataset (MQ sensor array,
DOI: 10.5281/zenodo.17285312) complemented by a physics-based 96-hour simulated
formalin-masking generator, since no public dataset provides labeled
chemically-masked samples."
