"""
Iterated prisoner's dilemma: prompts, move parsing, payoffs and fixed-strategy bots.

Standard payoffs (T=5 > R=3 > P=1 > S=0). The number of rounds is not shown
to the agents, so there is no known last round to defect on.
"""

import random
import re

COOPERATE, DEFECT = "C", "D"
INVALID = "?"  # Recorded in history when a completion has no parseable move

# (my payoff, their payoff) indexed by (my move, their move)
PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

SYSTEM_PROMPT = "You are playing a repeated game. Reply with a single letter: C or D."

RULES = (
    "You are playing a repeated game against one opponent. Each round you both choose "
    "C (cooperate) or D (defect) at the same time.\n"
    "- Both C: you get 3, they get 3\n"
    "- You C, they D: you get 0, they get 5\n"
    "- You D, they C: you get 5, they get 0\n"
    "- Both D: you get 1, they get 1\n"
    "The game lasts an unknown number of rounds. Your goal is to score as many points as possible.\n"
    "A ? in the history means that player gave no valid move (it counts as C, but they scored 0).\n"
)

_MOVE_PATTERN = re.compile(r"\b(C|D|cooperate|defect)\b", re.IGNORECASE)


def build_prompt(my_moves: list[str], their_moves: list[str]) -> str:
    """Prompt for the next move, given the history from this player's point of view."""
    lines = [RULES]
    if my_moves:
        lines.append("History so far:")
        for i, (mine, theirs) in enumerate(zip(my_moves, their_moves), start=1):
            lines.append(f"Round {i}: you {mine}, opponent {theirs}")
    else:
        lines.append("This is the first round.")
    lines.append(f"\nRound {len(my_moves) + 1}. Your move (C or D):")
    return "\n".join(lines)


def parse_move(text: str) -> str | None:
    """Extract C or D from a completion, or None if neither is present."""
    match = _MOVE_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)[0].upper()


def score_round(move_a: str | None, move_b: str | None) -> tuple[float, float]:
    """
    Payoffs for one round. An invalid move (None) is played as C but its
    player scores 0, so emitting gibberish is never better than cooperating.
    """
    pay_a, pay_b = PAYOFFS[(move_a or COOPERATE, move_b or COOPERATE)]
    return (0 if move_a is None else pay_a), (0 if move_b is None else pay_b)


# Fixed-strategy opponents: (my_moves, their_moves) -> move
BOTS = {
    "tft": lambda mine, theirs: DEFECT if theirs and theirs[-1] == DEFECT else COOPERATE,
    "alld": lambda mine, theirs: DEFECT,
    "allc": lambda mine, theirs: COOPERATE,
    "random": lambda mine, theirs: random.choice([COOPERATE, DEFECT]),
    "grudger": lambda mine, theirs: DEFECT if DEFECT in theirs else COOPERATE,
}
