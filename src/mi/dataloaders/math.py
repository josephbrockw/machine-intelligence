import json
import requests
from pathlib import Path


def load_math500_test(local_path="math500_test.json", save_copy=True):
    """
    If github repo stops working, download from Hugging Face

    from datasets import load_dataset
    dset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    """
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if save_copy:  # Saves a local copy
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data


