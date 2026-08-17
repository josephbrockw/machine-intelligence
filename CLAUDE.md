# machine-intelligence

A personal learning repo. I work through books, articles, and research papers to
build a real understanding of ML/AI.

**Primary goal: I learn.** Not to produce impressive code, not to finish fast.
If a choice comes down to "clever" vs. "understandable", pick understandable.

**Secondary goal:** end up with snippets and notebooks I can come back to later
as reference when I want to use a similar technique in a real project. So code
should be readable months from now, with enough comments/prose to explain the
*idea*, not just the mechanics.

## Structure

One directory per resource, numbered in the order I started it, plus one shared
package that isn't a resource:

```
01-ThinkBayes2/          # Allen Downey, Think Bayes 2e — the book's own notebooks
   notebooks/
      chapNN.ipynb       # worked through in place, exercises solved inline
      utils.py           # helpers that ship with the book
02-reasoning-model-from-scratch/
   02_pretrained_llm/
      01_setup.ipynb
src/mi/                  # shared code, importable from any notebook (see below)
   gpu.py
```

New resources get the next number: `02-<ShortName>/`, `03-<ShortName>/`, etc.
Papers/articles that are too small for a whole chapter tree can still take a
number and just hold a notebook or a markdown notes file.

Where a resource ships its own notebooks (like ThinkBayes2), I work directly in
them rather than rewriting from scratch — the diff against the original is the
record of what I did.

### Shared code lives in `src/mi/`

The numbered resource directories can never appear in an import statement —
`01-ThinkBayes2` starts with a digit and contains a hyphen, so it isn't a valid
Python identifier. Two rules follow:

- **A helper used by one resource's notebooks sits next to those notebooks.**
  Jupyter puts the notebook's own directory on `sys.path`, so `from utils import
  ...` just works, which is how ThinkBayes2's own `utils.py` is picked up.
- **A helper used across resources goes in `src/mi/`,** imported as
  `from mi.gpu import check_gpu_availability`. The root `pyproject.toml` declares
  a build backend, so `uv sync` installs this project into `.venv` in editable
  mode and `mi` resolves from anywhere, independent of where the kernel started.

No `sys.path.append("..")` — if an import needs path juggling to work, the code
is in the wrong place.

## Environment

Managed by [uv](https://docs.astral.sh/uv/), Python 3.14 (pinned in
`.python-version`; `requires-python` in `pyproject.toml` is only a floor), one
shared environment for the whole repo (`pyproject.toml` at the root).

```bash
uv sync                  # install/refresh the environment
uv run jupyter lab       # start JupyterLab
uv add <package>         # add a dependency (shared across all resources)
```

## Hardware

**Assume Apple Silicon unless I explicitly say otherwise** — M4 Pro, 64 GB
unified memory, macOS. There is no NVIDIA GPU on this machine.

Practical consequences, since most books and papers assume CUDA:

- The torch device is `mps` (`torch.backends.mps.is_available()`), never `cuda`.
  Don't write `.cuda()` or `device="cuda"`; pick the device once and pass it
  around. `mi.gpu.check_gpu_availability()` reports what's actually available.
- CUDA-only tooling is off the table: bitsandbytes, flash-attention, xformers,
  hand-written Triton kernels, `load_in_4bit` / `load_in_8bit` quantisation. If
  a resource I'm working through depends on one of these, say so up front and
  suggest what to do instead rather than letting me discover it mid-notebook.
- Some ops have no MPS kernel yet and will raise. `PYTORCH_ENABLE_MPS_FALLBACK=1`
  makes them fall back to CPU, at a speed cost worth mentioning when relevant.
- Unified memory means no separate VRAM budget, but 64 GB total is the ceiling
  for model + activations + everything else. Flag when something obviously won't
  fit, and prefer the smallest model that still demonstrates the idea — I'm
  reading these to understand the mechanism, not to train something competitive.

## Working with me

**Explain every change.** Before or alongside any edit, tell me:

1. **Why it's necessary** — what's wrong or missing right now.
2. **How it works** — the mechanism, in plain language. If it involves a concept
   I'm currently learning (a distribution, an update rule, a library API), spell
   that out rather than assuming I already have it.

This matters more than the change itself. An unexplained diff is worse than no
diff, because I can't learn from it.

Some things that follow from that:

- Don't hand me a finished exercise solution when the point of the exercise is
  for me to derive it. Explain the approach, point at the relevant idea, and let
  me write it — unless I explicitly ask for the answer.
- Don't silently "fix" my code while doing something else. If you spot a bug or
  a misunderstanding in what I wrote, say so and explain it, separately from
  whatever I asked for.
- Prefer small, obvious code over compact code. Intermediate variables with real
  names are good. So are comments that say *why*.
- If I've got a concept wrong, tell me directly. Being wrong quietly is the
  failure mode I'm trying to avoid.
- If something is genuinely beyond the scope of what I'm reading right now, it's
  fine to say "this is a rabbit hole, here's the one-line version" and move on.
