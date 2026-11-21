X_tr, y_tr = X[tr], y[tr]
X_va, y_va = X[va], y[va]
SENSOR_EMBED = 16  # telemetry feature vector length (counterpart of Fv)

def build_sensor_branch(n_in):
    inp = keras.Input(shape=(n_in,), name="sensor_input")
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(SENSOR_EMBED, activation="relu", name="sensor_embedding")(x)  # Fs feature vector
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(3, activation="softmax", name="state")(x)   # fresh / stale / chemically masked
    branch = keras.Model(inp, out, name="sensor_branch")
    feat = keras.Model(inp, branch.get_layer("sensor_embedding").output, name="sensor_feature_extractor")
    return branch, feat

sensor_branch, sensor_feature_extractor = build_sensor_branch(len(FEATURE_COLS))
sensor_branch.compile(optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
sensor_class_weights = {c: float(min(len(y_tr) / max((y_tr == c).sum(), 1) / 3.0, 25.0))
                        for c in (0, 1, 2)}   # inverse-frequency, capped to keep val stable
print("Sensor class weights:", {k: round(v, 2) for k, v in sensor_class_weights.items()})
hs = sensor_branch.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                       batch_size=BATCH_SENSOR, epochs=EPOCHS_SENSOR, verbose=2,
                       class_weight=sensor_class_weights,
                       callbacks=[keras.callbacks.TerminateOnNaN()])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(hs.history["loss"], label="train"); plt.plot(hs.history["val_loss"], label="val")
plt.title("Sensor branch — loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(hs.history["accuracy"], label="train"); plt.plot(hs.history["val_accuracy"], label="val")
plt.title("Sensor branch — accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_branch_training.png", dpi=150); plt.show()

y_pred = np.argmax(sensor_branch.predict(X_va, batch_size=BATCH_SENSOR, verbose=0), axis=1)
cm = confusion_matrix(y_va, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Stale", "Fresh", "Masked"]).plot(cmap="Blues")
plt.title("Sensor branch — Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_confusion_matrix.png", dpi=150); plt.show()
sensor_cr = classification_report(y_va, y_pred, target_names=["Stale", "Fresh", "Masked"],
                                  digits=3, zero_division=0, output_dict=True)
print(classification_report(y_va, y_pred, target_names=["Stale", "Fresh", "Masked"], digits=3,
                            zero_division=0))

# One-vs-rest ROC curves of the sensor branch on the held-out validation split
# (evaluation only).
y_prob = sensor_branch.predict(X_va, batch_size=BATCH_SENSOR, verbose=0)
Y_bin = label_binarize(y_va, classes=[0, 1, 2])
plt.figure(figsize=(4.9, 4.9))
for k, name in enumerate(["Stale", "Fresh", "Masked"]):
    fpr_k, tpr_k, _ = roc_curve(Y_bin[:, k], y_prob[:, k])
    plt.plot(fpr_k, tpr_k, lw=1.8, label=f"{name} (AUC = {auc(fpr_k, tpr_k):.4f})")
fpr_mi, tpr_mi, _ = roc_curve(Y_bin.ravel(), y_prob.ravel())
plt.plot(fpr_mi, tpr_mi, "k:", lw=1.2, label=f"micro-avg (AUC = {auc(fpr_mi, tpr_mi):.4f})")
plt.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8, label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Sensor branch \u2014 one-vs-rest ROC (validation)")
plt.legend(loc="lower right", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{OUT}/sensor_roc_curves.png", dpi=150); plt.show()
auc_per_class = {name: round(float(auc(*roc_curve(Y_bin[:, k], y_prob[:, k])[:2])), 4)
                 for k, name in enumerate(["Stale", "Fresh", "Masked"])}
macro_auc = float(roc_auc_score(Y_bin, y_prob, average="macro", multi_class="ovr"))
print(f"Sensor branch one-vs-rest AUC (macro) = {macro_auc:.4f} on {len(y_va):,} rows | per-class: {auc_per_class}")

# Consolidated evaluation-metrics summary for both branches (confusion matrices,
# precision/recall/F1, ROC AUCs).
eval_metrics = {
    "image_branch": {
        "n_val_images": int(len(val_idx)),
        "accuracy": float(img_cr["accuracy"]),
        "precision": {n: img_cr[n]["precision"] for n in ("Stale", "Fresh")},
        "recall": {n: img_cr[n]["recall"] for n in ("Stale", "Fresh")},
        "f1_score": {n: img_cr[n]["f1-score"] for n in ("Stale", "Fresh")},
        "confusion_matrix": cm_img.tolist(),
        "roc_auc": auc_img_branch},
    "sensor_branch": {
        "n_val_rows": int(len(y_va)),
        "accuracy": float(sensor_cr["accuracy"]),
        "precision": {n: sensor_cr[n]["precision"] for n in ("Stale", "Fresh", "Masked")},
        "recall": {n: sensor_cr[n]["recall"] for n in ("Stale", "Fresh", "Masked")},
        "f1_score": {n: sensor_cr[n]["f1-score"] for n in ("Stale", "Fresh", "Masked")},
        "confusion_matrix": cm.tolist(),
        "roc_auc_macro_ovr": macro_auc,
        "roc_auc_per_class": auc_per_class},
}
with open(f"{OUT}/evaluation_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)
print("evaluation_metrics.json saved (confusion matrices, precision/recall/F1, ROC AUCs)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, cr, names, title in [
        (axes[0], img_cr, ["Stale", "Fresh"], "Image branch"),
        (axes[1], sensor_cr, ["Stale", "Fresh", "Masked"], "Sensor branch")]:
    x = np.arange(len(names)); w = 0.26
    for k, (metric, label) in enumerate([("precision", "Precision"), ("recall", "Recall"),
                                         ("f1-score", "F1-score")]):
        vals = [float(cr[n][metric]) for n in names]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.09)
    ax.set_title(title + " \u2014 precision / recall / F1 (validation)")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/branch_metric_bars.png", dpi=150); plt.show()

sensor_branch.save(f"{OUT}/sensor_branch.keras")
sensor_feature_extractor.save(f"{OUT}/sensor_feature_extractor.keras")
print("Sensor branch saved (separate training done).")
gc.collect()                                            # release training temporaries before Part C
