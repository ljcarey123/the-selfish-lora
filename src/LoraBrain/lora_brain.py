import numpy as np
import random
import torch
import re
from config_loader import SYSTEM_PROMPT, VALID_ACTIONS, GRID_SYMBOLS, AGENT_POS
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnNewlineOrSpace(StoppingCriteria):
    """Stop generation when we encounter newline, space after a word, or punctuation."""

    def __init__(self, tokenizer, start_length: int) -> None:
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.stop_tokens = set(
            [
                (
                    tokenizer.encode("\n", add_special_tokens=False)[0]
                    if tokenizer.encode("\n", add_special_tokens=False)
                    else None
                ),
                (
                    tokenizer.encode(",", add_special_tokens=False)[0]
                    if tokenizer.encode(",", add_special_tokens=False)
                    else None
                ),
            ]
        )
        self.stop_tokens.discard(None)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        # Check if we've generated at least one token beyond the prompt
        if input_ids.shape[1] <= self.start_length:
            return False

        # Get the last generated token
        last_token = input_ids[0, -1].item()

        # Stop if we hit a stop token
        if last_token in self.stop_tokens:
            return True

        # Decode the generated text to check for word boundaries
        generated_ids = input_ids[0, self.start_length :]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        # Stop if we have a complete word (contains space or has length > 2)
        if " " in generated_text or len(generated_text.split()) > 1:
            return True

        return False


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

    def _format_grid(self, perception: np.ndarray, agent_on_resource: bool = False) -> str:
        """Formats the agent's perception grid into a string for the LLM prompt.

        Args:
            perception: 5x5 numpy array with symbolic characters (., R, W, etc.)
            agent_on_resource: Whether agent is standing on a resource

        Returns:
            Formatted grid string with agent at center (A or @)
        """
        grid_lines = []
        agent_r, agent_c = AGENT_POS  # (2, 2) - center of 5x5 grid

        for i, row in enumerate(perception):
            row_str = " ".join(str(cell) for cell in row)
            # The agent is always at the center (2, 2) in the 5x5 perception
            if i == agent_r:
                # Replace center position with A or @ (if on resource)
                symbol = GRID_SYMBOLS["on_resource"] if agent_on_resource else GRID_SYMBOLS["agent"]
                # Replace the character at position agent_c*2 (accounting for spaces)
                chars = list(row_str)
                chars[agent_c * 2] = symbol
                row_str = "".join(chars)
            grid_lines.append(row_str)
        return "\n".join(grid_lines)

    def _parse_llm_response(self, response: str) -> str:
        """Finds the chosen action in the LLM's response (expects lowercase snake_case)."""
        # Clean the response
        response_clean = response.strip().lower()

        # Check for exact matches first (most reliable)
        for action in self.ACTIONS:
            if action.lower() in response_clean:
                return action

        # If no match found, try case-insensitive word boundary search
        for action in self.ACTIONS:
            if re.search(r"\b" + re.escape(action) + r"\b", response, re.IGNORECASE):
                return action

        return None  # Return None if no action is found

    def choose_action(
        self,
        perception: np.ndarray,
        base_model,
        tokenizer,
        agent_id: str,
        energy: float,
        last_action: str = None,
        agent_on_resource: bool = False,
    ) -> str:
        with torch.no_grad():
            # 1. Inject the agent's unique LoRA weights into the base model
            for name, param in base_model.named_parameters():
                if name in self.weights:
                    param.data.copy_(self.weights[name])

        # 2. Create the prompt using the exact format from fine-tuning
        grid_representation = self._format_grid(perception, agent_on_resource)

        system_prompt = SYSTEM_PROMPT

        # Determine energy level string
        if energy < 0.3:
            energy_level = "low"
        elif energy < 0.7:
            energy_level = "medium"
        else:
            energy_level = "high"

        # Build the user prompt exactly as in fine-tuning
        user_prompt = f"""PERCEPTION:
{grid_representation}

Energy: {energy:.2f} ({energy_level})
Last action: {last_action if last_action else "None"}

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

        # Create stopping criteria to end generation early when action is complete
        start_length = model_inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList([StopOnNewlineOrSpace(tokenizer, start_length)])

        with torch.no_grad():
            outputs = base_model.generate(
                **model_inputs,
                max_new_tokens=10,
                temperature=0.7,  # Balanced: enough diversity for evolution, not too random
                do_sample=True,  # Enable sampling for diversity
                top_p=0.9,  # Nucleus sampling
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
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
