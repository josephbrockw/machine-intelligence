import torch

from mi.lm.generate import sample_response
from mi.lm.prompts import render_prompt
from mi.lm.scoring import reward_rlvr, sequence_logprob, reward_format


def compute_grpo_loss(
        model,
        tokenizer,
        example,
        device,
        num_rollouts=2,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        reward = reward_rlvr(text, example["answer"], partial_credit=0.5)
        logp = sequence_logprob(model, token_ids, prompt_len)

        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)

    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss  # TODO: Add a KL term here

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def compute_grpo_loss_with_kl(
        model,
        ref_model,
        tokenizer,
        example,
        device,
        num_rollouts=2,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        kl_coeff=0.02,
):
    assert num_rollouts >= 2
    roll_logps, roll_ref_logps, roll_rewards, samples = [], [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if kl_coeff:
            with torch.no_grad():
                ref_logp = sequence_logprob(ref_model, token_ids, prompt_len)
        else:
            ref_logp = None

        reward = reward_rlvr(text, example["answer"], partial_credit=0.5)
        roll_rewards.append(reward)


        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)

        if kl_coeff:
            roll_ref_logps.append(ref_logp)

        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)

    if kl_coeff:
        ref_logps = torch.stack(roll_ref_logps).detach()

    pg_loss = -(advantages.detach() * logps).mean()

    if kl_coeff:
        kl_loss = kl_coeff * torch.mean(logps - ref_logps)
    else:
        kl_loss = torch.tensor(0.0, device=logps.device)

    loss = pg_loss + kl_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def compute_grpo_loss_plus_format_reward(
        model,
        ref_model,
        tokenizer,
        example,
        device,
        num_rollouts=2,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        kl_coeff=0.02,
        format_reward_weight=1.0,
):
    assert num_rollouts >= 2
    roll_logps, roll_ref_logps, roll_rewards, samples = [], [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if kl_coeff:
            with torch.no_grad():
                ref_logp = sequence_logprob(ref_model, token_ids, prompt_len)
        else:
            ref_logp = None

        logp = sequence_logprob(model, token_ids, prompt_len)
        partial_credit = 0.5
        rlvr_reward = reward_rlvr(text, example["answer"], partial_credit=partial_credit)
        format_reward = reward_format(token_ids, prompt_len) if rlvr_reward > partial_credit else 0.0
        reward = rlvr_reward + format_reward_weight * format_reward


        roll_rewards.append(reward)
        roll_logps.append(logp)

        if kl_coeff:
            roll_ref_logps.append(ref_logp)

        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)

    if kl_coeff:
        ref_logps = torch.stack(roll_ref_logps).detach()

    pg_loss = -(advantages.detach() * logps).mean()

    if kl_coeff:
        kl_loss = kl_coeff * torch.mean(logps - ref_logps)
    else:
        kl_loss = torch.tensor(0.0, device=logps.device)

    loss = pg_loss + kl_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }