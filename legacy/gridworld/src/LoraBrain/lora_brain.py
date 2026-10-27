"""
LoraBrain - Container for per-agent LoRA weights.

This class holds the evolved genetic material (LoRA adapter weights) for each agent.
The actual inference is handled by InferenceEngine.
"""

import torch


class LoraBrain:
    """
    Container for per-agent LoRA weights (genetic material).

    Each agent has unique LoRA weights that modify the base LLM's behavior.
    These weights are evolved through genetic algorithms (crossover, mutation).
    """

    def __init__(self, peft_model):
        """
        Initialise with zero weights matching the PEFT model's LoRA layers.

        Args:
            peft_model: PEFT model with LoRA adapters
        """
        self.weights = {}
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.weights[name] = torch.zeros_like(param)

    def get_weights(self) -> dict[str, torch.Tensor]:
        """
        Get the brain's LoRA weights.

        Returns:
            Dict mapping parameter names to weight tensors
        """
        return self.weights

    def set_weights(self, new_weights: dict[str, torch.Tensor]):
        """
        Set the brain's LoRA weights (used during evolution).

        Args:
            new_weights: Dict of parameter names -> tensors
        """
        self.weights = new_weights
