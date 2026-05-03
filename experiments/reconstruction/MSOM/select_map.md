# `MSOM_select.py`

## load_results_dict(filepath)

Opens the pickle file with the MSOM results and returns the stored dictionary. Dictionary keys are the hyperparameter ttuples. Values contain metrics: qe, entropy, dead_neurons, and state, which hold weights, context_weights, bmu trajectories, and sequence lengths.

## results_to_dataframe(results_dict)

Converts the nested experiment dictionary into a flat  pandas dataframe, one row per experiment. Key is:

```python
(m, n, init, metric, kernel, alpha, beta, train_epochs)
```

For each run it extracts the quantization error, entropy, dead neuron count, best epoch we ended on, and state. Then it computes the derived quantities: 

```python
neurons = m * n
active_neurons = neurons - dead_neurons
dead_ratio = dead_neurons / neurons
norm_entropy = entropy / log(active_neurons) # if there is more than 1 active neuron, otherwise its 0
# this is a math.log so log_e(active_neurons)
```

The normalization is here because the raw entropy grows with map size. So a bigger map would "look better" just because it has more neurons.

## shortlist_maps(df, qe_tol=0.05, min_norm_entropy=0.70, max_dead_ratio=0.25)

This function makes a shortlist instead of just blindly taking the lowest quantization error. It only keeps the runs that are within the 5% of the best QE, at least moderately well-utilized (norm_entropy >= 0.7) and not way too sparse (dead_ratio <= 0.25).

After that it sorts the leftovers by fewer neurons, lower QE, lower dead ratio, and higher normalized entropy. 

We do all this because we dont want an oversized map or a half-dead one even if it is a good fit QE wise for example. 

## pareto_front(df)

Computes the non-dominant models across the four objectives of lower QE, fewer neurons, lower death rate, higher normalization entropy. A model is dropped if another one is no worse on all four and strictly better on at least one (its called pareto filtering).

## choose_best_compromise(df)

Turns each of the four metrics into a z-score and combines them into one weighted score:

```python
0.5 * qe_z
0.25 * neurons_z
0.15 * dead_ratio_z
-0.10 * norm_entropy_z
```

A lower score is better. So QE is the strongest factor in the score, map size second, dead neurons matter somewhat, and entropy gets subtracted because it helps.

## Example

If a config has:

```python
m = 3, n = 4 # 12 neurons
dead_neurons = 3
active_neurons = 9
dead_ratio = 3/12 = 0.25
entropy = 1.5

norm_entropy = 1.5 / log(9) = 1.5 / 2.197 = 0.683
```

This run would fai the default shortlist because 0.683 < 0.7, because of the filter the function applies.
