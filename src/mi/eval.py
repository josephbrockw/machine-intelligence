import json
import warnings
import torch
import time
from pathlib import Path

from mi.extraction import split_into_parts, normalize_text, equality_check, extract_final_candidate
from mi.lm.generate import generate_text_stream_concat
from mi.lm.prompts import render_prompt


def generate_stats(output_token_ids, tokenizer, start_time, end_time):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")

    for name, backend in (
        ("CUDA", getattr(torch, "cuda", None)),
        ("XPU", getattr(torch, "xpu", None))
    ):
        if backend is not None and backend.is_available():
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )

            if hasattr(backend, "synchronize"):
                backend.synchronize()

            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")

            backend.reset_peak_memory_stats()


def grade_answer(pred_text, gt_text):
    result = False

    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )
        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )

        if (
            gt_parts and pred_parts
            and len(gt_parts) == len(pred_parts)
        ):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )

    return result

def eta_progress_message(
        processed,
        total,
        start_time,
        show_eta=False,
        label="Progress"
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total}  |  ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)
    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0

    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"

    message = f"{progress}  |  ETA {eta}"
    return message.ljust(pad_width)



def evaluate_math500_stream(
        model, tokenizer, device, math_data,
        out_path=None,
        max_new_tokens=512,
        verbose=False
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])
            gen_text = generate_text_stream_concat(
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose
            )

            extracted = extract_final_candidate(
                gen_text
            )
            is_correct = grade_answer(
                extracted, row["answer"]
            )
            num_correct += int(is_correct)
            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)

            if verbose:
                print(
                    f"\n\n{'=' * 50}\n{progress_msg}\n"
                    f"{'=' * 50}\nExtracted: {extracted}\n"
                    f"Expected: {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-' * 50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc