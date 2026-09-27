# Evaluation and limits

## Main score

The evaluator scores each Source 1 record, then averages those scores. This is **macro F0.5**. F0.5 gives precision more weight than recall, so false matches have a larger effect than they would under F1. See [`evaluation.py`](../code/business_entity_resolution/src/evaluation.py).

For one Source 1 record:

- No true matches and no predicted matches scores 1. This is a correct singleton.
- A non-empty prediction for a singleton scores 0.
- A true match with no prediction scores 0.
- Otherwise, the score uses the overlap between predicted IDs and true IDs to calculate precision and recall, then combines them with beta 0.5.

`macro_f05` scores every Source 1 ID in the truth map. A missing prediction entry counts as an empty prediction.

## Candidate recall

The baseline also reports candidate-pair recall: the share of true pairs that appeared in candidate search. This is separate from F0.5. It is a hard ceiling on final pair recall because later address scoring cannot recover a pair that candidate search did not return.

## Interpreting reported results

Metric values in README files describe runs on particular supplied data. Treat them as historical reports, not as results from the current machine or data. Re-run evaluation to make a new claim, and state the dataset and split used.
