# Performance Assessment Workflow

The app now supports live performance assessment after every recognition upload.

## Runtime flow

1. User uploads an image for recognition.
2. Model returns prediction (`Known person` or `Unknown`) and distance.
3. App asks: "Is this prediction correct?"
4. User feedback is stored and the app immediately updates:
	- confusion matrix counts (`TP`, `FP`, `TN`, `FN`)
	- derived metrics
	- ROC curve and AUC

If feedback is skipped, metrics are unchanged.

## Label mapping used by the app

- Positive class: `Known` (belongs to enrolled database)
- Negative class: `Unknown`

From user confirmation:

- If user says **Yes**: predicted class is treated as ground truth.
- If user says **No**: ground truth is flipped.

## Metrics shown in UI

- Recall (TPR): $\frac{TP}{TP + FN}$
- False Positive Rate: $\frac{FP}{FP + TN}$
- Precision: $\frac{TP}{TP + FP}$
- Specificity (TNR): $\frac{TN}{TN + FP}$
- Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
- F1 score: $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
- ROC AUC: trapezoidal area under ROC curve

## ROC details

- X-axis: False Positive Rate
- Y-axis: True Positive Rate (Recall)
- Curve is recomputed from all collected feedback samples after each update
- Dashed diagonal indicates random baseline
