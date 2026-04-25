# MSOM_backwards_decode

The goal of this module is to **reconstruct plausible predecessor histories** for any neuron in a trained MSOM.

Given a target neuron, we ask:

> “Which previous neurons could have led to this neuron winning, according to the *full* MSOM merged distance `(1-α)‖x−w‖² + α‖C_t−c‖²`?”

We do **not** need the original training sequences — everything is decoded from the stored `weights` and `context_weights`.

---

## Mental picture (because its 3AM and I dont have caffeine, forgiveness)

MSOM map has the shape `(m, n, dim)`:

* `m` = number of rows in the map
* `n` = number of columns
* `dim` = how many numbers are in each prototype vector

Lets say:

* `m = 2`
* `n = 3`
* `dim = 2`

then the map is a `2 x 3` grid, and each neuron stores two vectors of length `2`:

- `W[j]`: the normal prototype for neuron `j`
- `C[j]`: the context prototype for neuron `j`

If the map is flattened, the total number of neurons is:

$$
N = m \cdot n
$$

and both `W` and `C` become arrays of shape `(N, dim)`.

A simple way to think about a neuron `j` is:

- `W[j]` = what the current input looks like when neuron `j` wins
- `C[j]` = what context tends to lead into neuron `j`

---

## What the reconstruction does

The backward decode does not try to recover the exact original input sequence. It builds a plausible chain of BMUs that could lead to a chosen target neuron. So what you get is a likely  **BMU history on the map** , not the literal raw sequence the model saw during training.

Given a flat target index `target_idx`, the decoder asks which earlier BMUs make this target neuron a plausible next winner under the MSOM distance rule?

### Step 1: flatten the trained map

The decoder first flattens the map into:

* `W` with shape `(N, dim)` for the main prototypes
* `C` with shape `(N, dim)` for the context prototypes

where `N = m * n`.

So each neuron has:

* a main prototype `W[i]`
* a context prototype `C[i]`

### Step 2: build the context that each possible predecessor would generate

If neuron `i` had been the previous BMU, then the next-step context would be

$$
G_i = (1 - \beta) W_i + \beta C_i
$$

In code:

```python
G = (1.0 - beta) * W + beta * C
```

This is the same rule the forward MSOM uses during training and evaluation:

$$
C_t = (1 - \beta) w_{\text{prev}} + \beta c_{\text{prev}}
$$

So the backward decoder is using the same temporal rule as the trained model.

### Step 3: precompute the two distance tables

The decoder builds two pairwise squared-distance tables.

The first is the distance between main prototypes:

$$
D^{\text{input}}_{j,k} = |W_j - W_k|^2
$$

The second is the distance between predecessor-generated contexts and stored context prototypes:

$$
D^{\text{ctx}}_{i,k} = |G_i - C_k|^2
$$

This is important because the decoder does **not** use context similarity alone. It uses the same merged logic as the forward MSOM.

### Step 4: score a predecessor for a chosen target

Now fix a target neuron `j`.

The decoder treats `W[j]` as the prototype of the current step and asks:

> if neuron `i` was the previous BMU, which neuron `k` would win next?

It computes the MSOM-style merged cost

$$
\text{cost}_{i,k}^{(j)} = (1-\alpha)|W_j - W_k|^2 + \alpha|G_i - C_k|^2
$$

In code:

```python
all_costs = (1.0 - alpha) * input_dists[target_j][None, :] + alpha * ctx_dists
```

For each possible predecessor `i`, this gives one whole row of costs over all possible next winners `k`.

Then the decoder finds the best possible next winner:

```python
winners = np.argmin(all_costs, axis=1)
best_cost = np.min(all_costs, axis=1)
```

### Step 5: measure how well the chosen target fits

The decoder then asks: how well does the chosen target `j` fit as that next state?

It computes

$$
\text{selfcost}_i = \alpha |G_i - C_j|^2
$$

then

$$
\text{gap}_i = \text{self\_cost} *i - \min_k \text{cost}* {i,k}^{(j)}
$$

and finally

$$
\text{score}_i = \text{self\_cost}_i + \text{margin\_weight} \cdot \text{gap}_i
$$

Lower scores are better.

A low score means two things:

1. the generated context from predecessor `i` is close to the target neuron's stored context `C[j]`
2. the target neuron `j` is competitive with other possible next winners

That second point matters. A predecessor is not considered good just because its generated context is close to `C[j]`. It also has to make `j` look like a reasonable winner under the full MSOM distance rule.

If `hard_self_consistent=True`, the decoder becomes stricter. In that case, any predecessor that does **not** make `j` the actual best next winner is rejected.

---

## Worked example

Here is a small example that follows the current code logic.

Assume:

$$
m = 2,\quad n = 2,\quad N = 4,\quad dim = 2
$$

and the flattened neurons are

$$
W[0] = [0, 0], \quad C[0] = [0.1, 0.0]
$$

$$
W[1] = [1, 0], \quad C[1] = [0.8, 0.1]
$$

$$
W[2] = [1, 1], \quad C[2] = [0.9, 0.8]
$$

$$
W[3] = [0, 1], \quad C[3] = [0.2, 0.9]
$$

Let

$$
\beta = 0.5
$$

Then each possible predecessor generates

$$
G_i = 0.5 W_i + 0.5 C_i
$$

so

$$
G[0] = [0.05, 0.00]
$$

$$
G[1] = [0.90, 0.05]
$$

$$
G[2] = [0.95, 0.90]
$$

$$
G[3] = [0.10, 0.95]
$$

Now suppose the target neuron is

$$
j = 2
$$

So the current-step prototype is

$$
W[2] = [1, 1]
$$

and the target context prototype is

$$
C[2] = [0.9, 0.8]
$$

Let

$$
\alpha = 0.5
$$

### Input-distance table for the target

We compare `W[2]` to every main prototype:

$$
|W[2] - W[0]|^2 = |(1,1) - (0,0)|^2 = 2
$$

$$
|W[2] - W[1]|^2 = |(1,1) - (1,0)|^2 = 1
$$

$$
|W[2] - W[2]|^2 = 0
$$

$$
|W[2] - W[3]|^2 = |(1,1) - (0,1)|^2 = 1
$$

So the target row of the input-distance table is

$$
[2,\ 1,\ 0,\ 1]
$$

### Context-distance row for predecessor 1

Now try predecessor `i = 1`, which generates

$$
G[1] = [0.90, 0.05]
$$

Compare that to every context prototype:

$$
|G[1] - C[0]|^2 = |[0.90, 0.05] - [0.1, 0.0]|^2 = 0.6425
$$

$$
|G[1] - C[1]|^2 = |[0.90, 0.05] - [0.8, 0.1]|^2 = 0.0125
$$

$$
|G[1] - C[2]|^2 = |[0.90, 0.05] - [0.9, 0.8]|^2 = 0.5625
$$

$$
|G[1] - C[3]|^2 = |[0.90, 0.05] - [0.2, 0.9]|^2 = 1.2125
$$

So for predecessor `1`, the context-distance row is

$$
[0.6425,\ 0.0125,\ 0.5625,\ 1.2125]
$$

### Merge the input and context terms

The decoder combines the two terms using `alpha = 0.5`:

$$
\text{cost} *{1,k}^{(2)} = 0.5 \cdot D^{\text{input}}* {2,k} + 0.5 \cdot D^{\text{ctx}}_{1,k}
$$

That gives

$$
k=0:\ 0.5 \cdot 2 + 0.5 \cdot 0.6425 = 1.32125
$$

$$
k=1:\ 0.5 \cdot 1 + 0.5 \cdot 0.0125 = 0.50625
$$

$$
k=2:\ 0.5 \cdot 0 + 0.5 \cdot 0.5625 = 0.28125
$$

$$
k=3:\ 0.5 \cdot 1 + 0.5 \cdot 1.2125 = 1.10625
$$

So the best next winner is

$$
k = 2
$$

That means predecessor `1` makes target neuron `2` a valid next winner.

### Turn that into a predecessor score

For predecessor `1`, the target-specific part is

$$
\text{self\_cost}_1 = \alpha |G[1] - C[2]|^2 = 0.5 \cdot 0.5625 = 0.28125
$$

The best merged cost in that row is also

$$
\text{best_cost}_1 = 0.28125
$$

So

$$
\text{gap}_1 = 0.28125 - 0.28125 = 0
$$

and with `margin_weight = 2.0` the final score is

$$
\text{score}_1 = 0.28125 + 2.0 \cdot 0 = 0.28125
$$

That is a good predecessor score.

### Compare several predecessors

If we repeat the same calculation for all possible predecessors, the final scores for target neuron `2` are:

$$
\text{score}_0 = 0.68125
$$

$$
\text{score}_1 = 0.28125
$$

$$
\text{score}_2 = 0.00625
$$

$$
\text{score}_3 = 0.33125
$$

So the best predecessor is neuron `2` itself, and the next best one is neuron `1`.

This is an important detail: in the current code, self-transitions are allowed by default because `allow_self=True`.

So by default, predecessor `2` is valid here.

If you want to forbid immediate self-transitions, you must call the decoder with:

```python
allow_self=False
```

Then the best remaining predecessor in this example would be neuron `1`.

---

## How the beam decoder uses these scores

`beam_decode_chains(...)` starts from the target index and grows a chain backwards.

At each decode step it:

1. takes the current front neuron in the chain
2. scores all possible predecessors for that neuron
3. keeps only the best few partial chains
4. repeats until the requested depth is reached

The width of that search is controlled by `beam_width`.

By default the decoder also:

* allows self-transitions with `allow_self=True`
* avoids cycles with `avoid_cycles=True`
* adds a start-state score with `add_start_score=True`

That start-state score uses a zero context vector, which matches the forward MSOM rule at sequence start.

## What the decoder needs as input

The main inputs are:

* `state`
* `m`, `n`
* `alpha`, `beta`
* `target_idx`
* `depth`
* `beam_width`

You can also control:

* `margin_weight`
* `hard_self_consistent`
* `allow_self`
* `avoid_cycles`
* `add_start_score`

The decoder works from the trained map state, not from the original raw training sequences. So it can reconstruct plausible BMU chains from the map's internal memory, but it cannot reproduce the exact original sequence that created them.

---

## What the output means

The decoder returns one or more candidate BMU chains.

Each chain is a sequence of flat neuron indices. These indices describe a plausible path through the map that could lead to the chosen target neuron.

A lower total score means the chain is more consistent with the learned map state. It does **not** mean the chain is guaranteed to be the true training history. It only means the chain is a better fit under the model's learned prototypes and context memory.

You can turn a decoded chain into a prototype sequence by mapping each neuron index back to its main prototype vector. That gives a sequence in feature space rather than a sequence of map positions.

So there are really two ways to look at a decoded result:

* as a path of BMUs on the map
* as a path of prototype vectors in feature space

## How to read the results

A decoded chain is easiest to interpret as a hypothesis.

If the top few decoded chains are very similar, the transition structure is probably stable around that target neuron.

If several chains have similar scores but different histories, the transition is likely ambiguous. In that case, the map does not strongly prefer one predecessor path over another.

Self-transitions can also appear. In the current decoder, they are allowed by default. So the best predecessor of a neuron can be the same neuron, unless self-transitions are explicitly disabled.

Validation helps here. A low validation error means the decoded prototype sequence is internally consistent with the same MSOM rule used during forward matching.

If stored BMU trajectories are available, decoded predecessors can also be compared against the observed predecessors seen in the data. That does not prove a chain is correct, but it helps show whether the decoded history agrees with the trajectories the model actually saw.

---

## Limits

This decoder does not reconstruct the original raw sequence. It reconstructs a plausible BMU history from the trained map state. So the result depends entirely on what the map stored in `weights` and `context_weights`.

That means the decoder inherits the strengths and weaknesses of the trained MSOM. If the map learned a clear transition pattern, the decoded chains can be informative. If the learned state is noisy or ambiguous, the decoded histories will be ambiguous too.

So the reconstruction should be read as a model-based explanation of what the map remembers, not as a literal replay of the training data.
