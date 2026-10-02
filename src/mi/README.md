# `mi` — shared code for the learning repo

Installed into `.venv` in editable mode by `uv sync`, so `import mi` resolves
from any notebook in any resource directory, wherever the kernel was started.

## Why anything is in here

The numbered resource directories can't appear in an import statement —
`01-ThinkBayes2` starts with a digit and contains a hyphen. So there are only
two places a helper can live:

- **next to the notebooks that use it**, picked up because Jupyter puts a
  notebook's own directory on `sys.path` (this is how ThinkBayes2's own
  `utils.py` is found);
- **here**, if anything else needs it.

Note the sharp edge in the first option: Jupyter adds the notebook's *own*
directory, not its parent. `03_evaluation/utils.py` is invisible from
`07_improving_grpo/`. So code shared across the **chapters of a single book**
also has to live here, even though it isn't shared across books. That's the
case `resources/` exists to handle.

Never `sys.path.append("..")`. If an import needs path juggling, the code is in
the wrong place.

## The one distinction that matters

Two kinds of code end up in `mi`, and they age differently. Sort by asking:

> **Would this still be correct against a different dataset, model, or book?**

| | answer | where it goes |
|---|---|---|
| **technique** | yes | a domain package — `lm/`, `rl/`, `bayes/` |
| **resource specifics** | no | `resources/<name>/` |

Resource specifics are the things one book takes for granted: dataset URLs and
JSON field names, prompt wording, a chapter's exact reward function, one
model's special-token ids, `\boxed{}` as an answer convention.

A useful tell — **if a function reaches into a dict by a literal key, or embeds
a URL, or quotes prose at the model, it's resource-specific.** Technique
functions take plain values (tensors, token ids, strings, floats) and stay
agnostic about where they came from.

The payoff is that the import line itself tells you which bucket something is
in, at the call site, months later:

```python
from mi.lm.sampling import top_p_filter                       # reusable
from mi.resources.reasoning_model_from_scratch.data import (  # this book only
    load_distill_data,
)
```

That's the point of the split. `mi.lm` is the part that survives into a real
project; `mi.resources` is scaffolding for working through one text.

## Layout

```
src/mi/
  gpu.py                 # device selection — generic
  lm/                    # language-model technique: generate, sample, score
  rl/                    # RL technique: the GRPO losses
  bayes/                 # Bayesian technique
  resources/
    reasoning_model_from_scratch/   # 02-reasoning-model-from-scratch
```

Domain packages are named for a **subject**, never a code-kind. `lm/`, `rl/`,
`bayes/` — not `loaders/`, `utils/`, `helpers/`. Organising by code-kind is what
produces "where does this go?" questions, because every file is plausibly a
helper.

## Adding a new resource

Name the subpackage after the resource directory: drop the leading number, turn
hyphens into underscores, lowercase.

| resource directory | subpackage |
|---|---|
| `01-ThinkBayes2` | `resources/think_bayes2/` |
| `02-reasoning-model-from-scratch` | `resources/reasoning_model_from_scratch/` |

Mechanical in both directions, so neither name has to be remembered. Give it an
`__init__.py` with a docstring naming the actual book or paper — the directory
name alone won't mean much in a year.

Only create one when a helper genuinely needs to be shared across that
resource's chapters. A helper used by exactly one notebook still belongs next to
that notebook.

## Where does a new helper go?

1. Used by one notebook only? → next to that notebook. Stop.
2. Correct against any dataset/model? → the matching domain package
   (`lm/`, `rl/`, `bayes/`), or a new subject-named one.
3. Otherwise → `resources/<name>/`.

When a function straddles 2 and 3 — generic mechanism, resource-specific inputs
— split it. Put the mechanism in the domain package taking plain arguments, and
let a thin function in `resources/` unpack the dataset record and call it.

New subpackages under `src/mi/` are picked up by the editable install without
re-running `uv sync`. If an import unexpectedly fails, `uv sync` is the fix.

## Current state (2026-09-28)

**This layout is the target, not yet the reality.** `resources/` is new and
empty apart from `__init__.py` files. The existing modules were all written
while working through book 02 and still mix both kinds of code — `lm/prompts.py`
holds that book's exact templates, `dataloaders/math.py` holds its dataset URLs,
`lm/scoring.py` mixes generic log-prob helpers with `reward_rlvr` and Qwen3's
hard-coded think-token ids, `eval.py` is MATH-500-shaped, and
`src/evaluate_math500.py` sits outside the package entirely.

Migration is deferred until book 02 is finished. Until then:

- **new code follows the rules above**, so the pile stops growing;
- **existing imports keep working** — nothing has moved.

`dataloaders/` is a code-kind, not a subject, and is slated to dissolve into the
relevant `resources/` packages. Don't add to it.
