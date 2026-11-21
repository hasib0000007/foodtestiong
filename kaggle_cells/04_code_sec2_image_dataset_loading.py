def discover_class_dirs(root, dataset_name):
    class_dirs = {}
    for d in sorted(glob.glob(os.path.join(root, "**/"), recursive=True)):
        d = d.rstrip(os.sep)
        files = [f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            class_dirs[os.path.join(dataset_name, os.path.relpath(d, root))] = d
        if len(class_dirs) >= 60:
            break
    return class_dirs

def _string_label(text):
    n = str(text).lower()
    if any(k in n for k in ("stale", "rotten", "discard", "spoil", "not fresh", "non-fresh",
                            "non fresh", "nonfresh", "half")):
        return 0
    if "fresh" in n:
        return 1
    return None

def freshness_label(class_name):
    """Checks only the class-folder basename, never the full path: full-path checks
    break on slugs such as 'fresh-and-stale-images...'."""
    return _string_label(os.path.basename(str(class_name)))

def labels_from_classes_csv(folder):
    """Parse a Roboflow multiclass export (one folder, images + _classes.csv with
    columns like 'filename,Fresh,Half-Fresh,Spoiled'). Returns [(filepath, 0/1)].
    Half-Fresh and Spoiled both count as not-fresh (label 0)."""
    pairs = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or len(df.columns) < 2:
            continue
        fname_col, cls_cols = df.columns[0], list(df.columns[1:])
        if not df[fname_col].astype(str).str.lower().str.endswith((".jpg", ".jpeg", ".png")).all():
            continue
        for _, r in df.iterrows():
            fp = os.path.join(folder, str(r[fname_col]))
            if not os.path.isfile(fp):
                continue
            best = max(cls_cols, key=lambda c: r[c])
            lab = _string_label(best)
            if lab is not None:
                pairs.append((fp, lab))
    return pairs

def _make_demo_images(root, per_class=12):
    """Tiny synthetic image set used ONLY for local dry runs (never on Kaggle)."""
    os.makedirs(root, exist_ok=True)
    for name, rgb in (("FreshApple", (150, 200, 60)), ("StaleApple", (90, 90, 30)),
                      ("FreshFish", (200, 190, 170)), ("StaleFish", (110, 105, 95))):
        d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
        for k in range(per_class):
            arr = np.clip(np.array(rgb) + np.random.normal(0, 25, (96, 96, 3)), 0, 255).astype("uint8")
            keras.utils.save_img(os.path.join(d, f"img_{k}.jpg"), arr)

paths, labels, dataset_of, provenance = [], [], [], []
for root in IMAGE_DATASETS:
    if not os.path.isdir(root):
        print(f"Image dataset not attached, skipping: {root}")
        continue
    dataset_name = os.path.basename(root.rstrip("/"))
    n0 = len(paths)
    for name, folder in discover_class_dirs(root, dataset_name).items():
        csv_pairs = labels_from_classes_csv(folder)   # Roboflow multiclass export (meat)
        if csv_pairs:
            for fp, lab in csv_pairs:
                paths.append(fp); labels.append(lab); dataset_of.append(dataset_name)
            continue
        # rel-path labeling: tilapia's fresh/stale info lives in PARENT folders
        # (Fresh_Day1/Eyes, NonFresh/Day3/Eyes), not in the leaf basename ('Eyes').
        lab = _string_label(os.path.relpath(folder, root))
        if lab is None:
            continue
        for f in glob.glob(os.path.join(folder, "*")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f); labels.append(lab); dataset_of.append(dataset_name)
    provenance.append({"dataset": dataset_name, "images": len(paths) - n0, "label": "fresh/stale"})

if not paths and not RUNNING_ON_KAGGLE:
    _make_demo_images("_demo_images")
    IMAGE_DATASETS = ["_demo_images"]
    for name, folder in discover_class_dirs("_demo_images", "_demo_images").items():
        lab = _string_label(os.path.basename(name))
        if lab is None:
            continue
        for f in glob.glob(os.path.join(folder, "*")):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f); labels.append(lab); dataset_of.append("_demo_images")
    provenance.append({"dataset": "_demo_images", "images": len(paths), "label": "fresh/stale"})

assert paths, ("No image datasets found — attach at least one Kaggle image dataset and add its "
               "path to IMAGE_DATASETS in the config cell.")
paths = np.array(paths); labels = np.array(labels); dataset_of = np.array(dataset_of)
for prov in provenance:
    print(f"  {prov['dataset']:55s} {prov['images']:6d} images")
print(f"TOTAL: {len(paths)} images (fresh={labels.sum()}, stale={(labels==0).sum()})")

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0          # rescale to [0,1]
    return img, label

def make_ds(indices, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths[indices], labels[indices]))
    if shuffle: ds = ds.shuffle(1024, seed=SEED)
    return ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

idx = np.random.permutation(len(paths))
n_val = max(1, int(len(paths) * IMG_VAL_FRACTION))
val_idx, tr_idx = idx[:n_val], idx[n_val:]
train_ds, val_ds = make_ds(tr_idx, True), make_ds(val_idx, False)
