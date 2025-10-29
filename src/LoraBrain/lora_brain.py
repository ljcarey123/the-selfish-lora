import numpy as np
import random
import torch
import re
from config_loader import SYSTEM_PROMPT, VALID_ACTIONS


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
        grid_lines = []
        for i, row in enumerate(perception):
            row_str = " ".join([tiles.get(val, "?") for val in row])
            # The agent is always at the center
            if i == 1:
                row_str = (
                    row_str[:2] + "A" + row_str[3:]
                )  # Replace middle character with 'A'
            grid_lines.append(row_str)
        return "\n".join(grid_lines)

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

        # 2. Create the prompt using the exact format from fine-tuning
        grid_representation = self._format_grid(perception)

        system_prompt = SYSTEM_PROMPT

        # Format history to match fine-tuning format (last 3 actions)
        history_formatted = action_history[-3:] if action_history else []
        # Pad with empty strings if less than 3
        while len(history_formatted) < 3:
            history_formatted.insert(0, "")

        history_str = "\n".join(f"- {h}" if h else "-" for h in history_formatted)

        # Build the user prompt exactly as in fine-tuning
        user_prompt = f"""PERCEPTION:
{grid_representation}

HISTORY (Last 3 actions):
{history_str}

ACTION:"""

        # 3. Get the LLM's decision using the chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # For inference, we use add_generation_prompt=True to signal the model to generate a response.
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Debug: Print the full prompt being sent to the model
        print("\n" + "="*80)
        print("DEBUG - FULL PROMPT:")
        print("-"*40)
        print(f"SYSTEM: {system_prompt}")
        print(f"USER: {user_prompt}")
        print("-"*40 + "\n")
        
        model_inputs = tokenizer(prompt_string, return_tensors="pt").to(
            base_model.device
        )
        with torch.no_grad():
            outputs = base_model.generate(
                **model_inputs,
                max_new_tokens=10,
                do_sample=False,  # Use greedy decoding for deterministic outputs
                pad_token_id=tokenizer.eos_token_id,
            )
        # For chat templates, we need to decode only the newly generated tokens
        response_ids = outputs[0][model_inputs.input_ids.shape[-1] :]
        newly_generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Debug: Print the raw model response
        print("DEBUG - MODEL RESPONSE:")
        print("-"*40)
        print(f"{newly_generated_text}")
        print("="*80 + "\n")

        # 4. Parse the response
        chosen_action = self._parse_llm_response(newly_generated_text)
        
        # Debug: Print the parsed action
        print(f"DEBUG - PARSED ACTION: {chosen_action}")
        print("="*80 + "\n")

        return chosen_action, newly_generated_text
