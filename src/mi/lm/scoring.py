import torch

from mi.extraction import extract_final_candidate
import math


def heuristic_score(
        answer,
        prompt=None,
        brevity_bonus=500.0,
        boxed_bonus=2.0,
        extract_bonus=1.0,
        fulltext_bonus=0.0,
):
    score = 0.0

    cand = extract_final_candidate(answer, fallback="none")

    if cand:
        score += boxed_bonus
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(answer, fallback="number_then_full")
            if cand:
                score += fulltext_bonus

    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score


@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device, show=True):
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    all_probas = torch.softmax(logits, dim=-1)

    t_idx = torch.arange(0, token_ids.shape[0] -1, device=device)
    next_ids = token_ids[1:]
    next_token_probas = all_probas[t_idx, next_ids]
    prod_next_token_probas = torch.prod(next_token_probas)

    if show:
        print("Next-token probabilities:", next_token_probas)
        print("Joint probability:", prod_next_token_probas)

    return next_token_probas, prod_next_token_probas


@torch.inference_mode()
def calc_next_token_logprobas(model, tokenizer, prompt, device, show=True):
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    all_logprobas = torch.log_softmax(logits, dim=-1)

    t_idx = torch.arange(0, token_ids.shape[0] -1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]

    sum_next_token_logprobas = torch.sum(next_token_logprobas)

    if show:
        print("Next token log-probabilities:", next_token_logprobas)
        print("Joint log-probability:", sum_next_token_logprobas)

    return next_token_logprobas, sum_next_token_logprobas


@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    return torch.mean(next_token_logps)
