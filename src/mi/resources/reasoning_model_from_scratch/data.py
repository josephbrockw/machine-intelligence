"""The datasets this book works with, and how to read a record out of them.

Everything here is specific to *Build a Reasoning Model (From Scratch)*: the
download URLs, the JSON field names, and the exact string format the student
model is trained to emit. That is why it lives under `mi.resources` rather than
in a generic module — see ../../README.md.

Currently holds the DeepSeek-R1 distillation data used in chapter 08
(`08_efficient_reasoning/distillation.ipynb`). Its two siblings,
`load_math500_test` and `load_math_train`, are still in `mi.dataloaders.math`
and belong here too; they move when the book is finished and the wider `mi`
migration happens.

A note on `format_distilled_answer`: it produces the *target* side of a
supervised distillation pair. The input side is `render_prompt_with_think_tokens`
(presently in `mi.lm.prompts`), and the two must agree on the `<think>`
convention exactly — the model only learns the format both halves share.
"""

import json
import requests
from pathlib import Path


import json
import requests
from pathlib import Path
from mi.resources.reasoning_model_from_scratch.prompts import render_prompt


def load_distill_data(
    local_path=None,
    partition="deepseek-r1-math-train",
    save_copy=True,
):

    if local_path is None:
        local_path = f"{partition}.json"
    local_path = Path(local_path, f"{partition}.json")

    url = (
        "https://huggingface.co/datasets/rasbt/math_distill"
        "/resolve/main/data/"
        f"{partition}.json"
    )
    backup_url = (
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        f"MATH/{partition}.json"
    )

    if local_path.exists(): #1
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        size_kb = local_path.stat().st_size / 1e3
        print(f"{local_path}: {size_kb:.1f} KB (cached)")
        return data

    assert partition in (
        "deepseek-r1-math-train",
        "deepseek-r1-math500",
        "qwen3-235b-a22b-math-train",
        "qwen3-235b-a22b-math500",
    )

    try:  #2
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        print("Using backup URL.")
        r = requests.get(backup_url, timeout=30)
        r.raise_for_status()

    data = r.json()

    if save_copy:  #3
        with local_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        size_kb = local_path.stat().st_size / 1e3
        print(f"{local_path}: {size_kb:.1f} KB")

    return data


def format_distilled_answer(entry):
    content = str(entry["message_content"]).strip()
    if not content:
        raise ValueError("Missing non-empty 'message_content' field.")

    thinking = str(entry["message_thinking"]).strip()
    return f"<think>{thinking}</think>\n\n{content}"


def build_examples(data, tokenizer):
    examples = []
    skipped = 0
    for entry in data:
        try:
            prompt = render_prompt(entry["problem"])
            prompt_ids = tokenizer.encode(prompt)

            target_answer = format_distilled_answer(entry)
            answer_ids = tokenizer.encode(target_answer, chat_wrapped=False)

            token_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]

            if len(token_ids) < 2:
                skipped += 1
                continue

            examples.append(
                {
                    "token_ids": token_ids,
                    "prompt_len": len(prompt_ids),
                }
            )
        except (KeyError, TypeError, ValueError):
            skipped += 1

    return examples, skipped