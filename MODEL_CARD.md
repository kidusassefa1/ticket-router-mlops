# Model Card — Ticket Router (Queue Classification)

## Model summary
This model routes customer support tickets into one of 52 queues using a fine-tuned transformer (XLM-R).

## Intended use
- Internal support routing assistant
- Helps triage tickets to the correct team faster

## Not intended use
- High-stakes decisions without human review
- Use as the only decision-maker for sensitive cases

## Data
Dataset: Tobi-Bueck/customer-support-tickets (multi-language support tickets)
Label: queue (52 labels)

## Metrics (best run)
- test_accuracy: 0.509
- test_f1_macro: 0.832
- test_f1_weighted: 0.509

## Known limitations
- Some queues overlap in meaning (e.g., “Technical Support” vs “IT Support”)
- Performance may vary by language and writing style
- Model may be overconfident on unclear tickets

## Ethics / Responsible AI notes
- Risk: bias across languages or domains
- Mitigation plan:
  - monitor performance by language
  - add human-in-the-loop review for low confidence predictions
  - log and review misrouted tickets regularly