"""
Batched greedy generation with whatever genome is currently loaded.

Load a genome once with genome.load_genome, then score all of its prompts in
a few batched calls. This is far cheaper than the legacy engine, which
reloaded weights and generated one prompt at a time.
"""

import torch


def generate(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str | None = None,
    max_new_tokens: int = 16,
    batch_size: int = 32,
) -> list[str]:
    """
    Generate a greedy completion for each prompt.

    Args:
        model: PEFT model with the desired genome already loaded
        tokenizer: Tokenizer with left padding
        prompts: User messages
        system_prompt: Optional system message prepended to each prompt
        max_new_tokens: Generation length cap
        batch_size: Prompts per forward pass

    Returns:
        Decoded completions (new tokens only), stripped
    """
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, batch["input_ids"].shape[1]:]
        outputs.extend(t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return outputs
