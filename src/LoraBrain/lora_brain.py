import numpy as np
import random
import torch
import re


class LoraBrain:
    """
    Represents the genetic material of an agent. It holds the LoRA weights
    that will be injected into the base LLM to guide its decisions.
    """

    ACTIONS = ["North", "South", "East", "West", "Gather"]

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

    def _create_contextual_perception(self, perception: np.ndarray) -> list[str]:
        """Translates the numeric grid perception into a list of natural language statements."""
        context_statements = []
        center = perception.shape[0] // 2

        # Cardinal directions for resources
        if perception[center, center + 1] == 1:
            context_statements.append("- A resource 'R' is to your East.")
        if perception[center, center - 1] == 1:
            context_statements.append("- A resource 'R' is to your West.")
        if perception[center - 1, center] == 1:
            context_statements.append("- A resource 'R' is to your North.")
        if perception[center + 1, center] == 1:
            context_statements.append("- A resource 'R' is to your South.")
        if perception[center, center] == 1:
            context_statements.append("- You are on top of a resource 'R'.")

        # Cardinal directions for walls
        if perception[center, center + 1] == -1:
            context_statements.append("- There is a wall to your East.")
        if perception[center, center - 1] == -1:
            context_statements.append("- There is a wall to your West.")
        if perception[center - 1, center] == -1:
            context_statements.append("- There is a wall to your North.")
        if perception[center + 1, center] == -1:
            context_statements.append("- There is a wall to your South.")

        return context_statements

    def _parse_llm_response(self, response: str) -> str:
        """Finds the chosen action in the LLM's response using a regex."""
        match = re.search(r"ACTION:\s*([A-Z_]+)", response.upper())
        if match:
            action = match.group(1)
            if action in self.ACTIONS:
                return action
        for action in self.ACTIONS:
            # Use regex to find whole word matches, case-insensitive
            if re.search(r'\b' + action + r'\b', response, re.IGNORECASE):
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

        # 2. Create the prompt
        context_list = self._create_contextual_perception(perception)

        # Conditionally add the 'Gather' action
        available_actions = [a for a in self.ACTIONS if a != "Gather"]
        # The center of the perception grid is the agent's current location
        center_x = center_y = perception.shape[0] // 2
        if perception[center_x, center_y] == 1:  # 1 represents a resource
            available_actions.append("Gather")

        context_list.append(f"- Available actions are {available_actions}.")
        if action_history:
            context_list.append(
                f"- Your recent history is: {action_history}. Avoid repeating pointless moves."
            )
        context_str = "\n".join(context_list)

        system_prompt = """You are a creature trying to survive. Your goal is to collect as many resources 'R' as possible. Choose the single best action from the list to achieve this goal."""

        user_prompt = f"""SITUATION:
{context_str}

ACTION:"""

        # 3. Get the LLM's decision
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_string = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
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
