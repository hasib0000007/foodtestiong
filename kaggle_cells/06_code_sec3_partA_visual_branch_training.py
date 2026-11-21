def build_image_branch():
    inp = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = layers.Rescaling(2.0, offset=-1.0, name="to_imagenet_range")(inp)  # [0,1] -> [-1,1] (ImageNet conv)
    if IMG_AUGMENT:   # active only during .fit(); disabled at inference/INT8 export
        x = layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = layers.RandomRotation(0.08, name="aug_rot")(x)
        x = layers.RandomZoom(0.08, name="aug_zoom")(x)
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), alpha=ALPHA, include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(EMBED_DIM, activation="relu", name="fv_embedding")(x)   # Fv feature vector
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="freshness")(x)
    branch = keras.Model(inp, out, name="image_branch")
    feat_model = keras.Model(inp, branch.get_layer("fv_embedding").output, name="image_feature_extractor")
    return branch, feat_model

image_branch, image_feature_extractor = build_image_branch()
image_branch.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss="binary_crossentropy", metrics=["accuracy"])
h1 = image_branch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2,
                      callbacks=[keras.callbacks.TerminateOnNaN()])

if EPOCHS_FINETUNE > 0:
    backbone = next(l for l in image_branch.layers if isinstance(l, keras.Model))  # MobileNetV2 backbone
    backbone.trainable = True
    for layer in backbone.layers[:-8]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):   # freeze BN stats (no val drop)
            layer.trainable = False
    image_branch.compile(optimizer=keras.optimizers.Adam(1e-5),
                         loss="binary_crossentropy", metrics=["accuracy"])
    ckpt = keras.callbacks.ModelCheckpoint(f"{OUT}/image_branch_best.keras",
                                           monitor="val_loss", save_best_only=True)
    h2 = image_branch.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, verbose=2,
                          callbacks=[keras.callbacks.TerminateOnNaN(), ckpt])
    image_branch = keras.models.load_model(f"{OUT}/image_branch_best.keras")   # keep best epoch
    image_feature_extractor = keras.Model(image_branch.input,
                                          image_branch.get_layer("fv_embedding").output,
                                          name="image_feature_extractor")
    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
else:
    hist = h1.history

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(hist["loss"], label="train"); plt.plot(hist["val_loss"], label="val")
plt.title("Image branch — loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(hist["accuracy"], label="train"); plt.plot(hist["val_accuracy"], label="val")
plt.title("Image branch — accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/image_branch_training.png", dpi=150); plt.show()

# ROC curve and AUC of the image branch on the held-out validation split (evaluation only).
img_prob = image_branch.predict(val_ds, verbose=0).ravel()
fpr_i, tpr_i, _ = roc_curve(labels[val_idx], img_prob)
auc_img_branch = float(auc(fpr_i, tpr_i))
plt.figure(figsize=(4.8, 4.8))
plt.plot(fpr_i, tpr_i, color="black", lw=1.8, label=f"image branch (AUC = {auc_img_branch:.4f})")
plt.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Image branch \u2014 ROC (validation)"); plt.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/image_branch_roc.png", dpi=150); plt.show()
print(f"Image branch validation AUC = {auc_img_branch:.4f} on {len(val_idx):,} held-out images")

# Precision/recall/F1 and the binary confusion matrix of the image branch
# (0.5 operating threshold, evaluation only).
img_pred = (img_prob >= 0.5).astype(int)
cm_img = confusion_matrix(labels[val_idx], img_pred, labels=[0, 1])
ConfusionMatrixDisplay(cm_img, display_labels=["Stale", "Fresh"]).plot(cmap="Blues")
plt.title("Image branch \u2014 Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/image_branch_confusion_matrix.png", dpi=150); plt.show()
img_cr = classification_report(labels[val_idx], img_pred, target_names=["Stale", "Fresh"],
                               digits=3, zero_division=0, output_dict=True)
print(classification_report(labels[val_idx], img_pred, target_names=["Stale", "Fresh"],
                            digits=3, zero_division=0))

image_branch.save(f"{OUT}/image_branch.keras")
image_feature_extractor.save(f"{OUT}/image_feature_extractor.keras")
print("Image branch saved (separate training done).")
