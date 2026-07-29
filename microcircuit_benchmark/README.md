# Microcircuit Benchmark

Tests whether a small recurrent unit with persistent state can outperform a conventional feed-forward MLP under a comparable parameter budget.

## Models
- **Matched MLP:** ordinary perceptron layers; the whole sequence is flattened.
- **Microcircuit network:** four small recurrent cells. Each keeps state across sequence steps and performs two local internal updates per input.

## Tasks
- 12-bit parity
- delayed XOR
- copy the first bit after distractors
- compositional sequence rule

## Robustness
- held-out train/validation/test splits
- five random seeds by default
- matched parameter counts
- wall-clock and approximate multiply-add counts
- mean and standard deviation
- automatic verdict with predefined thresholds

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
microbench --epochs 300 --seeds 5
```

Outputs appear in `results/`: `runs.csv`, `summary.csv`, `curves.csv`, `learning_curves.png`, and `report.md`.

A single toy-task win is not evidence. The hypothesis is provisionally supported only if the microcircuit wins by at least 3 percentage points on two tasks, including a delayed-memory task, across multiple seeds.
