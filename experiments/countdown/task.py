"""
Countdown puzzles from reasoning-gym: reach a target number with arithmetic
on a few given numbers. Scored by reasoning-gym's own checker:
1.0 correct, 0.05 a valid but wrong expression, 0.01 no answer.

Kept deliberately easy (3 small numbers, answer only, no reasoning) so
completions are short and the base 0.5B model scores above zero.
"""

import reasoning_gym as rg

SYSTEM_PROMPT = "You solve arithmetic puzzles. Reply with only the expression, nothing else."


class CountdownTask:
    def __init__(self, size: int, seed: int, num_numbers: int = 3, max_value: int = 20, max_target: int = 60):
        self.dataset = rg.create_dataset(
            "countdown",
            size=size,
            seed=seed,
            min_numbers=num_numbers,
            max_numbers=num_numbers,
            min_value=1,
            max_value=max_value,
            min_target=10,
            max_target=max_target,
        )
        self.entries = list(self.dataset)

    def prompts(self, indices: list[int]) -> list[str]:
        return [self.entries[i]["question"] for i in indices]

    def score(self, indices: list[int], completions: list[str]) -> list[float]:
        scores = []
        for i, text in zip(indices, completions):
            answer = text.splitlines()[0].strip().strip("`") if text else None
            scores.append(self.dataset.score_answer(answer, self.entries[i]))
        return scores
