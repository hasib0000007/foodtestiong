## 7. Edge Deployment Export (TFLite & C Headers)

The image branch is exported as a full-integer INT8 TensorFlow Lite model with
stratified calibration data, the fusion engine as a float32 TensorFlow Lite model
(TensorFlow Lite Micro-compatible), and the Mahalanobis detector parameters (mean,
shrunk inverse covariance, threshold) plus sensor scaling bounds as C header arrays
for the ESP32-S3 firmware.
