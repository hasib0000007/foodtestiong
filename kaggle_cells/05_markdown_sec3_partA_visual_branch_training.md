## 3. Part A: Visual Branch Training (MobileNetV2)

MobileNetV2 (alpha = 0.35) backbone with a compact sigmoid head, trained in two
phases: head training with the backbone frozen, then fine-tuning of the last
backbone blocks with BatchNorm statistics kept frozen. ROC/AUC,
precision/recall/F1 and the confusion matrix are reported on the held-out
validation split.
