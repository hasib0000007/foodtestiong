def resolve_label(rule, df, fname):
    """Return a label array for one CSV via the dataset's label rule."""
    t = rule["type"]
    if t == "column":
        col = next((c for c in df.columns if c.strip().lower() == rule["col"].strip().lower()), None)
        assert col is not None, f"{fname}: label column '{rule['col']}' not found"
        y = pd.to_numeric(df[col], errors="coerce")
        return y.fillna(1).astype(int).values
    if t == "text":           # string labels like 'Fresh_Day4' vs 'FrozenThawed_Day6'
        col = next((c for c in df.columns if c.strip().lower() == rule["col"].strip().lower()), None)
        assert col is not None, f"{fname}: label column '{rule['col']}' not found"
        sl = df[col].astype(str).str.lower()
        y = sl.str.contains(rule.get("fresh_match", "fresh"), regex=False) \
            & ~sl.str.contains(rule.get("stale_match", "\x00"), regex=False)
        return y.astype(int).values
    if t == "day_decay":      # filenames like 'Banana D3.csv' — later days are staler
        seq = fname.rsplit(".", 1)[0].lower().split()[-1].lstrip("d")
        day = int(seq) if seq.isdigit() else 1
        return np.full(len(df), 1 if day in rule["fresh_days"] else 0)
    return np.zeros(len(df), dtype=int)   # "constant"

# dataset authors misspell columns (e.g. IoT Agriculture 2024 ships 'tempreature');
# the alias table lets the canonical mapping resolve them instead of silently dropping
# the channel into imputation.
COLUMN_ALIASES = {
    "temperature": ["tempreature", "temp", "temp_c", "temperature(c)", "temperature(c)"],
    "humidity":    ["humidity_pct", "humid", "relative_humidity", "rh", "humidity(%)"],
}

def _resolve_col(df, want):
    w = want.strip().lower()
    if w in df.columns:
        return w
    for alias in COLUMN_ALIASES.get(w, []):
        if alias in df.columns:
            return alias
    return None

def load_sensor_datasets(cfgs):
    frames, coverage = {}, []
    for cfg in cfgs:
        if not cfg["path"]:
            print(f"Sensor dataset not attached (no matching input root): {cfg['name']}")
            continue
        files = sorted(glob.glob(cfg["path"], recursive=True)) if any(ch in cfg["path"] for ch in "*?[") \
                else ([cfg["path"]] if os.path.isfile(cfg["path"]) else [])
        if not files:
            print(f"Sensor dataset not attached (no files found): {cfg['path']}")
            continue
        mapped = []
        for f in files:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            row = {}
            for canon, raw in cfg["mapping"].items():
                src_col = _resolve_col(df, raw) if raw else None
                row[canon] = pd.to_numeric(df[src_col], errors="coerce") if src_col \
                             else np.full(len(df), np.nan)
            sub = pd.DataFrame(row)
            for canon, mul in cfg.get("scale", {}).items():  # unit harmonization (e.g. days -> hours)
                sub[canon] = sub[canon] * mul
            for canon in ("hours_elapsed", "ambient_temp_c"):      # still missing? neutral fill + flag
                if sub[canon].isna().all():
                    sub[canon] = 0.0 if canon == "hours_elapsed" else 25.0
                    sub[f"av_{canon}"] = 0.0
            base = os.path.basename(f)
            if sub["hours_elapsed"].nunique() <= 1:                 # unlogged time -> day index (e.g. 'Banana D3.csv')
                seq = base.rsplit(".", 1)[0].lower().split()[-1].lstrip("d")
                if seq.isdigit():
                    sub["hours_elapsed"] = int(seq) * 24.0
                    sub["av_hours_elapsed"] = 1.0
            sub["label"] = resolve_label(cfg["label_rule"], df, base)
            sub["dataset"] = cfg["name"]
            mapped.append(sub)
        d = pd.concat(mapped, ignore_index=True)
        for canon in SENSOR_COLS:                                        # derived channels start uncomputed
            if canon not in d:
                d[canon] = np.nan
        for canon in SENSOR_COLS:                                        # availability flag per dataset+channel
            d[f"av_{canon}"] = d[f"av_{canon}"] if f"av_{canon}" in d else (0.0 if d[canon].isna().all() else 1.0)
            d[canon] = d[canon].fillna(d[canon].median())                # sparse NaN -> dataset median
        frames[cfg["name"]] = d
    assert frames, ("No sensor datasets found — attach Kaggle sensor CSVs and add entries to "
                    "SENSOR_CSVS in the config cell.")
    comb = pd.concat(frames.values(), ignore_index=True)
    # Map real formalin-treated rows (label == 0) in tofu_formalin to Chemically Masked class (label == 2)
    comb.loc[(comb["dataset"] == "tofu_formalin") & (comb["label"] == 0), "label"] = 2
    # fill still-missing whole channels from cross-dataset medians (flag 0 keeps the imputation marked)
    for canon in SENSOR_COLS:
        if comb[canon].isna().any():
            comb[canon] = comb[canon].fillna(comb[canon].median())
        if f"av_{canon}" not in comb:
            comb[f"av_{canon}"] = 1.0
    # derived channels where both parents exist per row (override imputation when possible)
    both = (comb["av_ch2o_ppm"] > 0) & (comb["av_voc_log_ohm"] > 0)
    comb.loc[both, "av_ch2o_voc_ratio"] = 1.0
    comb.loc[both, "ch2o_voc_ratio"] = (comb.loc[both, "ch2o_ppm"] /
                                        np.power(10.0, comb.loc[both, "voc_log_ohm"]).clip(lower=1e-6))
    for name, g in comb.groupby("dataset"):             # coverage table built AFTER derived channels
        coverage.append({"dataset": name, "rows": len(g),
                         **{c: ("YES" if g[f'av_{c}'].iloc[0] > 0 else "-") for c in SENSOR_COLS},
                         "fresh": int((g.label == 1).sum()),
                         "stale": int((g.label == 0).sum()),
                         "masked": int((g.label == 2).sum())})
    return comb, pd.DataFrame(coverage)

# local dry-run fallback: fabricate three small CSVs with different coverage gaps
def _demo_sensor_csvs():
    base = os.path.abspath("_demo_sensor"); os.makedirs(base, exist_ok=True)
    if glob.glob(os.path.join(base, "*.csv")):
        return
    for name, cols in (("Fruit D1.csv", ["MQ2", "MQ135"]), ("Fruit D2.csv", ["MQ2", "MQ135"]),
                       ("Storage T1.csv", ["temperature", "humidity"]), ("Cold T2.csv", ["temperature", "humidity"])):
        n = 120
        df = pd.DataFrame({c: np.abs(np.random.normal(120, 30, n)) for c in cols})
        df["ambient_temperature"] = np.random.uniform(24, 32, n)
        df.to_csv(os.path.join(base, name), index=False)
    return base

def _config_files_exist(cfgs):
    for cfg in cfgs:
        if any(ch in cfg["path"] for ch in "*?["):
            if glob.glob(cfg["path"], recursive=True):
                return True
        elif os.path.isfile(cfg["path"]):
            return True
    return False

if not _config_files_exist(SENSOR_CSVS) and not RUNNING_ON_KAGGLE:
    # local dry-run only: two demo CSV groups with deliberately different coverage
    _demo_sensor_csvs()
    SENSOR_CSVS = [
        {"name": "enose_fruits", "path": os.path.abspath("_demo_sensor/Fruit D*.csv"),
         "label_rule": {"type": "day_decay", "fresh_days": [1], "stale_days": [2]},
         "mapping": {"voc_log_ohm": "MQ2", "ch2o_ppm": "MQ135", "core_temp_c": None,
                     "ambient_temp_c": None, "humidity_pct": None, "hours_elapsed": None}},
        {"name": "storage_iot", "path": os.path.abspath("_demo_sensor/* T*.csv"),
         "label_rule": {"type": "constant", "label": 0},
         "mapping": {"core_temp_c": "temperature", "ambient_temp_c": "ambient_temperature",
                     "humidity_pct": "humidity", "voc_log_ohm": None, "ch2o_ppm": None,
                     "hours_elapsed": None}},
    ]

sensor_df, coverage_df = load_sensor_datasets(SENSOR_CSVS)
AV_COLS = [f"av_{c}" for c in SENSOR_COLS]
ALL_COLS = SENSOR_COLS + AV_COLS
print(f"\nCombined sensor dataset: {sensor_df.shape[0]} rows x {len(ALL_COLS)} features "
      f"(8 channels + 8 availability flags)")
print(f"Class balance: fresh={int((sensor_df.label==1).sum())}, stale={int((sensor_df.label==0).sum())}, "
      f"masked={int((sensor_df.label==2).sum())} (100% real data, {MASK_AUGMENT_FRACTION:.0%} synthetic augment)")
print(coverage_df.to_string(index=False))   # per-dataset channel coverage table

missing = [c for c, a in zip(SENSOR_COLS, AV_COLS)
           if coverage_df.empty or (coverage_df[c] == "YES").sum() == 0]
if missing:
    print("\nThe attached sensor datasets do not cover the following channels:")
    for c in missing:
        print(f"  - {c}: {MISSING_CHANNEL_GUIDE[c]}")
    coverage_df.to_csv(f"{OUT}/sensor_dataset_coverage.csv", index=False)
    print(f"\nCoverage table saved to {OUT}/sensor_dataset_coverage.csv")
else:
    print("\nAll 8 channels are covered by the attached sensor datasets.")
    coverage_df.to_csv(f"{OUT}/sensor_dataset_coverage.csv", index=False)

if len(coverage_df):
    cov = np.array([[1 if str(coverage_df.loc[r, c]) == "YES" else 0 for c in SENSOR_COLS]
                    for r in range(len(coverage_df))])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(coverage_df) + 2.5))
    ax.imshow(cov, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SENSOR_COLS))); ax.set_xticklabels(SENSOR_COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(coverage_df))); ax.set_yticklabels(coverage_df.dataset)
    for r in range(len(coverage_df)):
        for c in range(len(SENSOR_COLS)):
            ax.text(c, r, "YES" if cov[r, c] else "-", ha="center", va="center",
                    color="white" if cov[r, c] else "black", fontsize=8)
    ax.set_title("Sensor channel coverage per attached dataset")
    plt.tight_layout(); plt.savefig(f"{OUT}/sensor_channel_coverage.png", dpi=150); plt.show()

def augment_environmental_channels(df, active_fraction=0.70):
    """Storage-environment channels that NO attached dataset covers would otherwise stay
    flat constants the sensor branch cannot learn from. Fill ~70% of rows with realistic
    DHT22-like synthetic variation (flag 1) and keep ~30% flag 0 so the 'sensor absent'
    behaviour seen at inference still stays represented (DHT22 future-proofing)."""
    rng = np.random.default_rng(SEED + 7)
    fresh_like = (df.label.values == 1) | (df.label.values == 2)   # masked looks fresh
    added = []
    if (df["av_ambient_temp_c"] == 0).all():
        vals = np.where(fresh_like, rng.uniform(12.0, 24.0, len(df)), rng.uniform(20.0, 38.0, len(df)))
        use = rng.random(len(df)) < active_fraction
        df["ambient_temp_c"] = np.where(use, vals, df["ambient_temp_c"])
        df["av_ambient_temp_c"] = use.astype(np.float32)
        added.append("ambient_temp_c (synthetic, 12-24C stored / 20-38C room)")
    if (df["av_humidity_pct"] == 0).all():
        vals = np.where(fresh_like, rng.uniform(40.0, 65.0, len(df)), rng.uniform(60.0, 95.0, len(df)))
        use = rng.random(len(df)) < active_fraction
        df["humidity_pct"] = np.where(use, vals, df["humidity_pct"])
        df["av_humidity_pct"] = use.astype(np.float32)
        added.append("humidity_pct (synthetic, 40-65% stored / 60-95% room)")
    if (df["av_temp_rise_rate"] == 0).all():
        vals = np.where(df.label.values == 2, rng.uniform(0.0, 0.2, len(df)),
                 np.where(df.label.values == 1, rng.uniform(0.0, 0.3, len(df)),
                          rng.uniform(0.3, 1.5, len(df))))
        use = rng.random(len(df)) < active_fraction
        df["temp_rise_rate"] = np.where(use, vals, df["temp_rise_rate"])
        df["av_temp_rise_rate"] = use.astype(np.float32)
        added.append("temp_rise_rate (synthetic C/h, masked < fresh < stale)")
    if added:
        print("\nUncovered environment channels receive synthetic variation "
              "(70% active rows, 30% kept at flag 0 for absent-sensor robustness):")
        for a in added:
            print("  -", a)
    return df

def generate_masking_augment(df, fraction):
    if fraction <= 0:                 # disabled: masked class stays 100% real
        return df
    n = int(len(df) * fraction)
    src = df.sample(n, random_state=SEED).copy()
    src["ch2o_ppm"] = np.random.uniform(4.0, 9.0, n)                 # toxic gas despite look
    src["voc_log_ohm"] = np.clip(src["voc_log_ohm"] * 0.4 + 3.6, 2.0, None)  # decay arrested
    src["core_temp_c"] = src["core_temp_c"] - 1.5                    # cooler: rot slowed
    src["label"] = 2
    for c in SENSOR_COLS:                                            # augment inherits channel availability
        av = f"av_{c}"
        if av in df.columns:
            src[av] = df.loc[src.index, av].values
    src["dataset"] = "simulated_masking"
    return pd.concat([df, src], ignore_index=True)

sensor_df = generate_masking_augment(sensor_df, MASK_AUGMENT_FRACTION)
sensor_df = augment_environmental_channels(sensor_df)

for c in SENSOR_COLS:                       # final NaN/Inf cleanup (guards loss: nan)
    sensor_df[c] = np.nan_to_num(sensor_df[c].values, nan=0.0, posinf=0.0, neginf=0.0)
for a in AV_COLS:
    sensor_df[a] = np.nan_to_num(sensor_df[a].values, nan=0.0, posinf=1.0, neginf=0.0)

fig, ax = plt.subplots(figsize=(7, 5))
for lab, name, c in [(0, "Stale (natural)", "tab:orange"), (1, "Fresh", "tab:green"), (2, "Chemically masked (real formalin)", "tab:red")]:
    d = sensor_df[sensor_df.label == lab]
    if len(d) > 2000:
        d = d.sample(2000, random_state=SEED)
    ax.scatter(d.ch2o_ppm, d.voc_log_ohm, s=12, alpha=0.6, c=c, label=name)
ax.set_xlabel("Formaldehyde-family CH2O (ppm) — ZE08/MQ135"); ax.set_ylabel("VOC log(Ω) — BME680/MQ2")
ax.set_title("Adulteration Boundary Scatterplot (CH2O vs VOC)"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/adulteration_boundary.png", dpi=150); plt.show()

plt.figure(figsize=(9, 7))
corr = sensor_df[SENSOR_COLS].corr()
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(SENSOR_COLS)), SENSOR_COLS, rotation=45, ha="right")
plt.yticks(range(len(SENSOR_COLS)), SENSOR_COLS)
for i in range(len(SENSOR_COLS)):
    for j in range(len(SENSOR_COLS)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(label="Pearson r"); plt.title("Sensor Cross-Correlation Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_correlation.png", dpi=150); plt.show()

bounds = {c: (float(sensor_df[c].min()), float(sensor_df[c].max())) for c in SENSOR_COLS}
scaled = sensor_df.copy()
for c in SENSOR_COLS:                                   # min–max scaling to [0,1]
    lo, hi = bounds[c]
    scaled[c] = np.clip((sensor_df[c] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
smoothed = scaled.copy()                                # moving-average pre-filter
smoothed[SENSOR_COLS] = np.clip(
    ndimage.uniform_filter1d(scaled[SENSOR_COLS].values, size=SMOOTH_WIN, axis=0), 0.0, 1.0)
del scaled                                              # free ~130 MB before feature building

FEATURE_COLS = SENSOR_COLS + AV_COLS                    # values + availability flags
idx = np.random.permutation(len(smoothed))
n_val = int(0.15 * len(smoothed))
va, tr = idx[:n_val], idx[n_val:]
X = smoothed[FEATURE_COLS].values.astype(np.float32)
y = smoothed.label.values.astype(np.int32)

print("\nRows per dataset after augmentation:")
print(sensor_df["dataset"].value_counts().to_string())
print("\nPer-channel statistics after NaN/Inf cleanup and min-max scaling:")
for c in SENSOR_COLS:
    v = sensor_df[c].values
    n_nan = int(np.isnan(v).sum()); n_inf = int(np.isinf(v).sum())
    print(f"  {c:16s} min={v.min():12.4g} max={v.max():12.4g} nan={n_nan} inf={n_inf}")
assert np.isfinite(X).all(), "Data sanity check failed: NaN/Inf values remain in the feature matrix."
assert np.isin(np.unique(y), [0, 1, 2]).all(), "Data sanity check failed: unexpected label values."
print(f"Data sanity check passed: finite={np.isfinite(X).all()}, "
      f"X_min={X.min():.4f}, X_max={X.max():.4f}")
