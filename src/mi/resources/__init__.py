"""Code that belongs to one specific book, paper, or article.

Each resource gets a subpackage named after its directory at the repo root,
minus the leading number and with hyphens turned into underscores
(`02-reasoning-model-from-scratch` -> `reasoning_model_from_scratch`).

Anything in here is free to hard-code what a single resource takes for granted:
dataset URLs and JSON field names, prompt wording, the exact reward used in one
chapter, a specific model's token ids. That is the whole point -- it keeps those
assumptions out of the generic modules (`mi.lm`, `mi.rl`, `mi.bayes`), which
stay usable by the next resource. See ../README.md for the full rule.
"""
