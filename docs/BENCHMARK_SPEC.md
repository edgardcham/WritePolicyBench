# WritePolicyBench Specification v0

**Focus**: Document/API drift under memory write budgets

## 1. Episode Schema

### 1.1 Step Structure

```python
@dataclass(frozen=True)
class Step:
    t: int                          # Timestep index (0-indexed)
    observation: Any                # Payload (dict, str, object)
    metadata: dict[str, Any]        # Optional annotations
```

**Example**:
```python
Step(
    t=42,
    observation={
        "api": "stripe.v2023_10.charge.create",
        "params": ["amount", "currency", "source"],
        "deprecated_at": "2024-01-15"
    },
    metadata={
        "source": "stripe_docs",
        "version_distance": 2,
        "breaking_change": True
    }
)
```

### 1.2 Episode Structure

```python
@dataclass(frozen=True)
class Episode:
    steps: list[Step]               # Sequential stream (immutable)
    labels: dict[str, Any]          # Ground truth / eval targets
```

**Example labels**:
```python
{
    "critical_steps": [5, 12, 29],      # Must-remember indices
    "deprecated_apis": 7,                # Count of deprecated items
    "breaking_changes": [12, 29],        # High-priority writes
    "total_drift_events": 15
}
```

### 1.3 Observation Types

- **API Schema**: `{"endpoint": str, "params": list, "deprecated": bool}`
- **Documentation**: `{"section": str, "content": str, "last_updated": date}`
- **Migration Guide**: `{"old_api": str, "new_api": str, "deadline": date}`

---

## 2. Memory Item Schema

### 2.1 Stored Item

```python
@dataclass
class MemoryItem:
    step: Step                      # Original step
    written_at: int                 # Timestep when written
    byte_cost: int                  # Storage cost
    metadata: dict[str, Any]        # Tags, priority, etc.
```

### 2.2 Byte Accounting Assumptions

**Base calculation**:
```python
def estimate_bytes(step: Step) -> int:
    # JSON serialization + overhead
    payload = len(json.dumps(step.observation))
    metadata = len(json.dumps(step.metadata))
    header = 32  # t, pointers, struct overhead
    return payload + metadata + header
```

**Default assumptions**:
- Empty step: 32 bytes (header only)
- Typical API observation: 200–500 bytes
- Rich documentation: 1–5 KB
- Per-item index overhead: 16 bytes (timestep, offset)

**Budget units**: Bytes (not step count), enforced at write time.

---

## 3. Write Actions

### 3.1 Action Types

| Action   | Description                                      | Cost Formula              |
|----------|--------------------------------------------------|---------------------------|
| `SKIP`   | Ignore step, no write                            | 0 bytes                   |
| `WRITE`  | Write full step to memory                        | `estimate_bytes(step)`    |
| `MERGE`  | Combine with existing item (delta only)          | `bytes(delta) + 16`       |
| `EXPIRE` | Remove old item, free budget                     | `-original_cost`          |

### 3.2 Constraints

**Write**:
- Must have `remaining_budget >= estimate_bytes(step)`
- Item is append-only (no in-place edits)

**Merge** (delta-only, reviewer-proof):
- **Target must exist** and must be a **base WRITE item** (MERGE-to-MERGE chains are invalid).
- **Same-endpoint only**: for dict observations, both the incoming and target observations must contain an `"api"` field and they must be **exactly equal**.
- **Delta schema**: `delta` is a **shallow dict patch** containing only changed fields from the incoming observation **excluding** the primary key field `"api"`.
  - Canonical delta: `delta[k] = new_observation[k]` iff `k != "api"` and `new_observation[k] != base_observation[k]`.
  - `delta` must be **non-empty** (no-op MERGE is rejected to prevent inflating the retained set at near-zero cost).
  - If a policy supplies `delta`, it must match the canonical delta exactly; otherwise the action is rejected.
- **Cost**: `bytes(json(delta)) + 16` (fixed merge metadata overhead).
- Requires `remaining_budget >= merge_cost`.

**Expire**:
- Item must exist and be older than current timestep
- Frees exact byte cost from original write
- Budget is credited immediately

**Budget overflow**: If action exceeds budget, it is rejected (policy must handle gracefully).

### 3.3 MemoryAction Schema

**Canonical action dataclass** (from `memory.py`):
```python
@dataclass(frozen=True)
class MemoryAction:
    action: ActionType                  # "SKIP" | "WRITE" | "MERGE" | "EXPIRE"
    step: Step | None = None            # Required for WRITE, MERGE
    target_t: int | None = None         # Required for MERGE, EXPIRE
    delta: dict[str, Any] | None = None # Optional for MERGE
    reason: str | None = None           # Optional debug info
```

**Example actions**:
```python
# Write a critical step
MemoryAction(action="WRITE", step=incoming_step, reason="breaking_change")

# Merge new data into existing item at t=45
MemoryAction(action="MERGE", step=incoming_step, target_t=45, delta={"new_param": "email"})

# Expire old item to free budget
MemoryAction(action="EXPIRE", target_t=10, reason="LRU eviction")

# Skip (no-op)
MemoryAction(action="SKIP")
```

---

## 4. Metrics

### 4.1 Success Metrics

Let $R$ be the set of relevant timesteps (e.g., `labels["critical_steps"]`).
Let $W$ be the set of **retained timesteps** induced by the final memory contents.

**Retained timesteps $W$ (WRITE vs. MERGE)**:
- A stored **WRITE** item at timestep `t` contributes `t` to $W$.
- A stored **MERGE** delta item at timestep `t` contributes `t` to $W$ **only if**:
  1) its `merge_parent_t` refers to a **base WRITE** item that is still present in final memory, and
  2) the episode's endpoint identity at `t` matches the endpoint identity at `merge_parent_t` (same `observation["api"]`).

If the parent base is expired, the delta becomes an *orphan* and does **not** count toward $W$.

**Recall @ Budget**:
```
recall = |W ∩ R| / |R|
```

**Precision @ Budget**:
```
precision = |W ∩ R| / |W|
```

**F1 Score**:
```
f1 = 2 * (precision * recall) / (precision + recall)
```

### 4.2 Utility Metrics

**Utility-per-KB**:
```
utility_per_kb = total_utility(W) / (bytes_used / 1024)
```
- `total_utility(W)` = sum of the **episode-provided** per-timestep utility labels for `t ∈ W`
- Higher is better (efficiency)

**Regret (WRITE-only oracle, clamped)**:
```
regret = max(0, U*(B) - U(W))
```
- The oracle $U^*(B)$ is computed under the **WRITE-only** byte estimator (`estimate_bytes`).
- When MERGE is enabled, policies can legitimately outperform this baseline via delta storage; we therefore **clamp regret to be non-negative** for interpretability.

### 4.3 Staleness Metrics

**Average Staleness**:
```
staleness(t) = current_t - t
avg_staleness = mean([staleness(t) for t in W])
```

**Drift Coverage**:
```
drift_coverage = |W ∩ labels["critical_steps"]| / labels["total_drift_events"]
```

**Expired Item Rate**:
```
expire_rate = count(EXPIRE actions) / count(WRITE actions)
```

### 4.4 Budget Efficiency

**Budget Utilization**:
```
utilization = bytes_used / max_budget
```

**Write Density**:
```
write_density = |W| / len(episode.steps)
```

---

## 5. Baseline Policies

### 5.1 Trivial Baselines

1. **AlwaysWrite**: Write every step until budget exhausted
2. **NeverWrite**: Skip all steps (0% recall, 0 cost)
3. **UniformSample**: Write every Nth step uniformly

### 5.2 Heuristic Baselines

4. **PriorityThreshold**: Write if `step.metadata["priority"] > threshold`
5. **RecencyBias**: LRU-style, expire oldest when budget full
6. **UtilityGreedy**: Write highest `utility` steps first (oracle-sorted)

### 5.3 Learning Baselines

7. **RandomPolicy**: Random write with probability `p`
8. **EpsilonGreedy**: Exploit utility estimates, explore with ε
9. **BanditUCB**: Upper-confidence-bound arm selection per step type

### 5.4 Advanced Baselines

10. **MergeAggressive**: Prefer MERGE over WRITE when possible
11. **ExpireOldest**: Proactively expire items older than `T` timesteps
12. **OracleOptimal**: Hindsight-optimal (cheating baseline, upper bound)

---

## 6. Experiment Protocol

### 6.1 Frozen Episode Stream

**Requirement**: All experiments use identical episodes for reproducibility.

**Format**:
```python
episodes = load_jsonl("episodes/api_drift_v1.jsonl")
# Each line: {"steps": [...], "labels": {...}}
```

**Episode sources**:
- Scraped API changelogs (Stripe, Twilio, GitHub)
- Documentation version diffs (Python, React, TensorFlow)
- Synthetic drift generators (configurable drift rates)

### 6.2 Budget Grid

**Primary budgets** (bytes):
- **Tiny**: 1 KB (≈5 API steps)
- **Small**: 10 KB (≈50 steps)
- **Medium**: 100 KB (≈500 steps)
- **Large**: 1 MB (≈5000 steps)

**Episode lengths**:
- Short: 100 steps
- Medium: 1000 steps
- Long: 10,000 steps

**Grid**: All combinations (4 budgets × 3 lengths = 12 conditions)

### 6.3 Evaluation Loop

**Policy interface**: `select(step: Step, store: ByteMemoryStore) -> Iterable[MemoryAction]`

Policies can emit **multiple actions** per step (e.g., EXPIRE old items, then WRITE new).

```python
for episode in episodes:
    for budget_size in [1024, 10240, 102400, 1048576]:  # 1KB, 10KB, 100KB, 1MB
        budget = ByteBudget(max_bytes=budget_size)
        store = ByteMemoryStore(budget=budget)
        policy = initialize_policy()

        for step in episode.steps:
            # Policy returns iterable of actions (can be empty, single, or multiple)
            actions = policy.select(step, store)

            for action in actions:
                success = store.apply(action)  # Store handles budget enforcement
                if not success:
                    # Action rejected (e.g., insufficient budget)
                    pass

        metrics = evaluate(store.items(), episode.labels, budget)
        results.append(metrics)
```

**Example policy emitting multiple actions**:
```python
def select(self, step: Step, store: ByteMemoryStore) -> Iterable[MemoryAction]:
    # Expire oldest item if budget tight
    if store.budget.remaining() < 500 and (oldest := store.oldest_item()):
        yield MemoryAction(action="EXPIRE", target_t=oldest.step.t)

    # Write new step if critical
    if step.metadata.get("breaking_change"):
        yield MemoryAction(action="WRITE", step=step, reason="critical")
```

### 6.4 Reporting

**Per-policy outputs**:
- `recall_at_budget.csv`: Recall vs. budget (varying budgets)
- `regret_curve.png`: Cumulative regret over timesteps
- `utility_per_kb.json`: Efficiency by episode type
- `confusion_matrix.csv`: Precision/recall breakdown

**Aggregated leaderboard**:
```
Policy             | Recall@10KB | F1@10KB | Util/KB | Avg Staleness
-------------------|-------------|---------|---------|---------------
OracleOptimal      | 0.95        | 0.94    | 12.3    | 5.2
UtilityGreedy      | 0.78        | 0.71    | 9.1     | 12.5
PriorityThreshold  | 0.62        | 0.58    | 6.4     | 18.3
UniformSample      | 0.45        | 0.41    | 4.2     | 25.1
```

---

## 7. Implementation Checklist

- [ ] Episode loader (`load_jsonl`)
- [ ] Byte accounting (`estimate_bytes`)
- [ ] Budget enforcement (`Budget.consume`)
- [ ] Action executor (WRITE/MERGE/EXPIRE)
- [ ] Metrics calculator (recall, precision, utility-per-KB, staleness)
- [ ] Baseline policies (12 total)
- [ ] Experiment runner (budget × episode grid)
- [ ] Results aggregator + leaderboard generator

---

## 8. Extensions (Future Work)

- **Hierarchical budgets**: Separate budgets for API vs. docs
- **Write latency**: Penalize slow writes (async budget)
- **Compression**: Allow policies to compress observations
- **Retrieval tasks**: Query memory for specific APIs (retrieval@k)
- **Multi-agent**: Competing policies sharing budget
- **Adversarial drift**: Worst-case drift patterns

---

**Version**: 0.1
**Last Updated**: 2026-01-29
**Status**: Implementable draft (ready for harness development)