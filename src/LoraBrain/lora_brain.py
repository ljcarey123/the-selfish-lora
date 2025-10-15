import numpy as np
import random
import torch
import re
from .config_loader import SYSTEM_PROMPT, VALID_ACTIONS


class LoraBrain:
    """
    Represents the genetic material of an agent. It holds the LoRA weights
    that will be injected into the base LLM to guide its decisions.
    """

    ACTIONS = VALID_ACTIONS

    def __init__(self, peft_model):
        # Initialize with random weights that match the PEFT model's layers
        self.weights = {}
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.weights[name] = torch.zeros_like(param)

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Returns the brain's current LoRA weights."""
        return self.weights

    def set_weights(self, new_weights: dict[str, torch.Tensor]):
        """Sets the brain's LoRA weights."""
        self.weights = new_weights

    def _format_grid(self, perception: np.ndarray) -> str:
        """Formats the agent's perception grid into a string for the LLM prompt."""
        # Mapping from grid values to characters
        tiles = {0: ".", 1: "R", -1: "W"}
        grid_str = "+---+\n"
        for i, row in enumerate(perception):
            row_str = "".join([tiles.get(val, "?") for val in row])
            # The agent is always at the center
            if i == 1:
                row_str = row_str[:1] + "A" + row_str[2:]
            grid_str += f"|{row_str}|\n"
        grid_str += "+---+"
        return grid_str

    def _parse_llm_response(self, response: str) -> str:
        """Finds the chosen action in the LLM's response using a regex."""
        match = re.search(r"ACTION:\s*([A-Z_]+)", response.upper())
        if match:
            action = match.group(1)
            if action in self.ACTIONS:
                return action
        for action in self.ACTIONS:
            # Use regex to find whole word matches, case-insensitive
            if re.search(r"\b" + action + r"\b", response, re.IGNORECASE):
                return action
        return None  # Return None if no action is found

    def choose_action(
        self,
        perception: np.ndarray,
        base_model,
        tokenizer,
        agent_id: str,
        action_history: list[str],
    ) -> str:
        with torch.no_grad():
            # 1. Inject the agent's unique LoRA weights into the base model
            for name, param in base_model.named_parameters():
                if name in self.weights:
                    param.data.copy_(self.weights[name])

        # 2. Create the prompt using the new grid-based format
        grid_representation = self._format_grid(perception)

        system_prompt = SYSTEM_PROMPT
        # Conditionally add the 'Gather' action to the list of available actions
        available_actions = [a for a in self.ACTIONS if a != "Gather"]
        center_x = center_y = perception.shape[0] // 2
        if perception[center_x, center_y] == 1:  # 1 represents a resource
            available_actions.append("Gather")

        # Build the user prompt with all available context
        prompt_parts = [f"SITUATION:\n{grid_representation}"]
        prompt_parts.append(f"- Available actions are {available_actions}.")
        if action_history:
            prompt_parts.append(f"- Your recent history is: {action_history}.")
        prompt_parts.append("\nACTION:")

        user_prompt = "\n".join(prompt_parts)

        # 3. Get the LLM's decision using the chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # For inference, we use add_generation_prompt=True to signal the model to generate a response.
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer(prompt_string, return_tensors="pt").to(
            base_model.device
        )
        with torch.no_grad():
            outputs = base_model.generate(**model_inputs, max_new_tokens=10)
        # For chat templates, we need to decode only the newly generated tokens
        response_ids = outputs[0][model_inputs.input_ids.shape[-1] :]
        newly_generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # 4. Parse the response
        chosen_action = self._parse_llm_response(newly_generated_text)

        return chosen_action, newly_generated_text
