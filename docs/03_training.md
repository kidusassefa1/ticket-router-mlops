# Training

## Task
Multi-class text classification (ticket → queue). Current setup uses 52 queue labels.

## Model
- XLM-R (XLM-RoBERTa base) fine-tuned with Hugging Face Trainer (PyTorch)

## Experiments
Baseline:
- 1 epoch
- test_accuracy ~0.49
- test_f1_macro ~0.68

Improved:
- class-weighted loss (handles imbalance)
- 3 epochs
- test_accuracy 0.509
- test_f1_macro 0.832

## Why class weights
With many queues, the dataset is imbalanced.
Class weights increase loss for underrepresented queues so the model learns them better.
This improved macro F1 significantly.