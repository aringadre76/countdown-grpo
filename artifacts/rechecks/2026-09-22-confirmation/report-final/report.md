# Frozen confirmation comparison

Greedy and sampled pass@4 are task-level measurements. Paired percentile
bootstrap intervals use 10,000 task resamples with seed 20260918.
Aggregate intervals condition on these three observed training runs; they
do not measure uncertainty over the population of possible training seeds.

| Checkpoint | Suite | Mode | Initialization | Trained | Paired gain | 95% interval |
|---|---|---|---:|---:|---:|---|
| seed-42 | fresh_confirmation | greedy | 0.39% | 9.38% | 8.98% | [5.47%, 12.50%] |
| seed-42 | fresh_confirmation | sample | 0.39% | 14.84% | 14.45% | [10.16%, 19.14%] |
| seed-42 | source_confirmation | greedy | 0.00% | 17.58% | 17.58% | [12.89%, 22.27%] |
| seed-42 | source_confirmation | sample | 1.17% | 34.77% | 33.59% | [27.34%, 39.84%] |
| seed-43 | fresh_confirmation | greedy | 0.39% | 7.42% | 7.03% | [3.91%, 10.55%] |
| seed-43 | fresh_confirmation | sample | 0.39% | 15.23% | 14.84% | [10.55%, 19.53%] |
| seed-43 | source_confirmation | greedy | 0.00% | 21.09% | 21.09% | [16.02%, 26.17%] |
| seed-43 | source_confirmation | sample | 1.17% | 35.94% | 34.77% | [28.91%, 40.62%] |
| seed-44 | fresh_confirmation | greedy | 0.39% | 8.59% | 8.20% | [4.69%, 11.72%] |
| seed-44 | fresh_confirmation | sample | 0.39% | 16.80% | 16.41% | [11.72%, 21.09%] |
| seed-44 | source_confirmation | greedy | 0.00% | 16.80% | 16.80% | [12.50%, 21.48%] |
| seed-44 | source_confirmation | sample | 1.17% | 35.94% | 34.77% | [28.91%, 40.62%] |

## Three-seed aggregate

This is the mean of three seed-specific task outcomes. The interval resamples tasks; it is not uncertainty over the training-seed population.

| Models | Suite | Mode | Initialization | Mean trained | Paired gain | 95% interval |
|---|---|---|---:|---:|---:|---|
| three-seed mean | fresh_confirmation | greedy | 0.39% | 8.46% | 8.07% | [5.08%, 11.20%] |
| three-seed mean | fresh_confirmation | sample | 0.39% | 15.62% | 15.23% | [11.33%, 19.40%] |
| three-seed mean | source_confirmation | greedy | 0.00% | 18.49% | 18.49% | [14.32%, 22.92%] |
| three-seed mean | source_confirmation | sample | 1.17% | 35.55% | 34.37% | [28.91%, 39.97%] |

## Greedy comparison after equality-suffix normalization

The diagnostic removes only a terminal `= integer` from extracted expressions, then applies the unchanged verifier.
It checks whether that simple formatting repair is sufficient to explain the gain.

| Models | Suite | Initialization | Mean trained | Paired gain | 95% interval |
|---|---|---:|---:|---:|---|
| three-seed mean | fresh_confirmation | 1.95% | 8.46% | 6.51% | [3.26%, 9.77%] |
| three-seed mean | source_confirmation | 1.56% | 18.49% | 16.93% | [12.37%, 21.48%] |

## Fixed-order trace audit

Each seed includes up to 20 greedy primary-solve disagreements selected by task ID across both suites, including losses.
Repeated tasks across seeds are repeated seed-task comparisons, not independent puzzles.

| Seed | Changed legal arithmetic reaches target | Formatting repair sufficient | Ambiguous | Lost solution |
|---|---:|---:|---:|---:|
| seed-42 | 6 | 1 | 12 | 1 |
| seed-43 | 5 | 1 | 13 | 1 |
| seed-44 | 4 | 1 | 14 | 1 |

The audit supports a narrow arithmetic-search interpretation where a legal wrong-target expression changes to a legal exact expression.
Ambiguous illegal-to-valid changes do not establish arithmetic search by themselves.

Normalized comparisons and solvable-only results are in comparison.json.
Audit JSONL includes both gains and losses selected in fixed task-ID order.
