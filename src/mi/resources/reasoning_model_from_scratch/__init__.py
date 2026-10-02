"""Sebastian Raschka, *Build a Reasoning Model (From Scratch)*.

Repo directory: `02-reasoning-model-from-scratch/`

Holds the specifics of this book's setup -- the MATH / MATH-500 datasets and the
DeepSeek-R1 distillation data, the book's prompt templates, its reward
functions, and its Qwen3-0.6B evaluation loop.

Why this lives under `mi` rather than next to the notebooks: the book's chapter
directories cannot import from each other. Jupyter puts only a notebook's *own*
directory on sys.path, so `03_evaluation/utils.py` is invisible from
`07_improving_grpo/`. A helper shared across chapters therefore has to sit in
the installed package, even though it is specific to one resource.
"""
