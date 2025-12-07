# AUV-Path-Planning

This is official source code of *AUV Path Planning Considering Ocean Current Disturbance Based on Cloud Desktop Technology*(**Under Review!**)

The core algorithm code of the N-DDQNP model and its application in two environments.

## Multi-Objective Path Evaluation API

Based on the paper "Multi-objective path planning for unmanned surface vehicle with currents effects" (Yong Ma et al., ISA Transactions 75, 2018), we have implemented a unified evaluation module that computes four key objectives for USV path planning:

1. **Path Length (L0)** – total Euclidean distance of the path.
2. **Path Smoothness (θ0)** – maximum turning angle between consecutive segments (degrees).
3. **Economic Cost (E0)** – energy consumption considering currents and fuel consumption per unit time.
4. **Safety (D_Safe)** – aggregated safety degree based on distances to obstacles.

### Usage

The evaluation API is encapsulated in `usv_evaluation.py`. The main class is `MultiObjectiveEvaluator`.

#### Basic Example

```python
from usv_evaluation import MultiObjectiveEvaluator

# Create evaluator (optionally provide environment for current data)
evaluator = MultiObjectiveEvaluator(env=env, usv_length=10.0, usv_speed=6.0, fuel_consumption=1.0)

# Path as list of (x, y) points
path = [(5.0, 5.0), (50.0, 50.0), (80.0, 80.0)]
metrics = evaluator.evaluate(path)

print(f"Path Length: {metrics['path_length']:.2f}")
print(f"Smoothness: {metrics['smoothness']:.2f}°")
print(f"Economic Cost: {metrics['economic_cost']:.2f}")
print(f"Safety: {metrics['safety']:.2f}")
```

#### Integration with HTD3 Evaluation

The evaluation has been integrated into `HTD3_main.py`'s `evaluate_htd3` function. When running evaluation with `--do_eval`, the four metrics are automatically computed, printed, and saved to `multi_objective_results.csv` in the evaluation output directory.

To run a full evaluation with a trained model:

```bash
python HTD3_main.py --do_eval --output_dir htd3_output --eval_episodes 5
```

### File Structure

- `usv_evaluation.py` – contains the `MultiObjectiveEvaluator` class and helper functions.
- `HTD3_main.py` – modified evaluation function that uses the evaluator.
- `test_evaluation.py` – standalone test of the evaluator.
- `test_integration.py` – integration test with a trained model.

### Parameters

The evaluator constructor accepts:
- `env` (optional): an instance of `MyUsvEnv` to access ocean current data.
- `usv_length`: length of the USV (used for safety distance calculation).
- `usv_speed`: constant speed of the USV relative to water (V).
- `fuel_consumption`: fuel consumption per unit time (FCPUT).

Adjust these parameters according to your USV specifications and environment scaling.

### Output

The evaluator returns a dictionary with keys:
- `'path_length'` (L0)
- `'smoothness'` (θ0)
- `'economic_cost'` (E0)
- `'safety'` (D_Safe)

When integrated with HTD3, the results are saved as a CSV with columns:
`Episode, Reward, Length, UsvPathLength, TotalPathLength, PathLength, Smoothness, EconomicCost, Safety`.

### References

Yong Ma, Yanchao Zhao, Weidong Zhang, “Multi-objective path planning for unmanned surface vehicle with currents effects”, ISA Transactions, Volume 75, 2018, Pages 137-156, ISSN 0019-0578, https://doi.org/10.1016/j.isatra.2018.02.019.
