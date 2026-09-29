# Track A - Model registry status

Model: `TelcoChurnClassifier`

| version | stage | run_id | status | creation_timestamp |
| --- | --- | --- | --- | --- |
| 1 | Production | 13a66838ef43479981a710ed49f0b2e5 | READY | 2026-09-29T04:58:59.015000 |

## Recorded stage transitions (from training run)

```json
{
  "model_name": "TelcoChurnClassifier",
  "version": "1",
  "run_id": "13a66838ef43479981a710ed49f0b2e5",
  "run_name": "logreg_C10",
  "selection_metric": "f1",
  "selection_value": 0.6031746031746031,
  "transitions": [
    {
      "model": "TelcoChurnClassifier",
      "version": "1",
      "to_stage": "Staging",
      "from_run": "13a66838ef43479981a710ed49f0b2e5",
      "final_stage": "Staging"
    },
    {
      "model": "TelcoChurnClassifier",
      "version": "1",
      "to_stage": "Production",
      "from_run": "13a66838ef43479981a710ed49f0b2e5",
      "final_stage": "Production"
    }
  ]
}
```