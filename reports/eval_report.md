# Evaluation Report

## Best run metrics
- test_accuracy: 0.509
- test_f1_macro: 0.832
- test_f1_weighted: 0.509
- num_labels: 52

## Key observations
- Macro F1 improved significantly after class-weighted loss + 3 epochs
- Worst class in baseline (Technical Support) improved after weighting
- Confusion mostly happens between similar queues (support vs product vs customer service)

## Artifacts
- confusion_matrix.png
- errors.csv (top misclassified examples)
- class_weights.csv (if logged)