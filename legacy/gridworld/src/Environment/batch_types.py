"""
Batch inference types for vLLM-ready architecture.

This module defines the data structures for batch processing that enable
efficient multi-LoRA inference in a single forward pass.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class BatchInferenceRequest:
    """
    Request structure for batch LLM inference.

    This enables vLLM or other batch inference engines to process
    multiple agents with different LoRA adapters in a single forward pass.

    Attributes:
        prompts: List of text prompts (one per agent)
        adapter_ids: List of LoRA adapter identifiers (one per agent)
        agent_ids: List of agent IDs for result mapping
        metadata: Additional context (energy levels, positions, etc.)
    """
    prompts: List[str]
    adapter_ids: List[str]
    agent_ids: List[str]
    metadata: Dict[str, Any]

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.prompts)

    def validate(self) -> bool:
        """Validate that all lists have matching lengths."""
        lengths = [len(self.prompts), len(self.adapter_ids), len(self.agent_ids)]
        return len(set(lengths)) == 1  # All same length


@dataclass
class BatchInferenceResponse:
    """
    Response structure from batch LLM inference.

    Attributes:
        actions: List of action strings (parsed from LLM outputs)
        raw_outputs: List of raw LLM response strings
        agent_ids: List of agent IDs (same order as request)
        success: List of booleans indicating parse success
    """
    actions: List[str]
    raw_outputs: List[str]
    agent_ids: List[str]
    success: List[bool]

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary mapping agent_id -> (action, raw_output, success)."""
        return {
            agent_id: {
                "action": action,
                "raw_output": raw,
                "success": success
            }
            for agent_id, action, raw, success in zip(
                self.agent_ids, self.actions, self.raw_outputs, self.success
            )
        }


@dataclass
class BatchPerception:
    """
    Vectorized perception data for a batch of agents.

    Attributes:
        grids: (N, 5, 5) array of perception grids (integer encoded)
        agent_ids: List of agent IDs
        positions: (N, 2) array of agent positions
        energies: (N,) array of agent energy levels
        on_food: (N,) boolean array indicating if agent is on food
    """
    grids: np.ndarray  # Shape: (N, 5, 5)
    agent_ids: List[str]
    positions: np.ndarray  # Shape: (N, 2)
    energies: np.ndarray  # Shape: (N,)
    on_food: np.ndarray  # Shape: (N,) dtype=bool

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.agent_ids)

    def validate(self) -> bool:
        """Validate array shapes."""
        n = len(self.agent_ids)
        return (
            self.grids.shape == (n, 5, 5) and
            self.positions.shape == (n, 2) and
            self.energies.shape == (n,) and
            self.on_food.shape == (n,)
        )
