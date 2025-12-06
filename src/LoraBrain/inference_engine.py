"""
Inference Engine for multi-agent LLM processing with per-agent LoRA weights.

This module handles sequential inference for multiple agents, loading each agent's
unique LoRA weights before generating their action.
"""

import torch
import re
import time
from typing import Optional
from Environment.batch_types import BatchInferenceRequest, BatchInferenceResponse


class InferenceEngine:
    """
    Inference engine for multi-agent LLM processing with per-agent LoRA weights.

    This engine processes agents sequentially, loading each agent's unique LoRA
    weights before inference. This ensures each evolved agent uses its own
    adapted behaviour rather than shared weights.

    The engine handles:
    - Per-agent LoRA weight loading
    - Tokenisation and generation for each agent
    - Action parsing from LLM outputs
    - Error handling with fallback actions
    """

    def __init__(self, base_model, tokenizer, agent_brains: dict = None):
        """
        Initialise inference engine.

        Args:
            base_model: Base PEFT model with LoRA
            tokenizer: Tokenizer
            agent_brains: Optional dict mapping agent_id -> LoraBrain for weight loading
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.agent_brains = agent_brains or {}

        # Action parsing regex
        self.action_pattern = re.compile(r'\b(MOVE_NORTH|MOVE_SOUTH|MOVE_EAST|MOVE_WEST)\b', re.IGNORECASE)

        # Diagnostics
        self.inference_count = 0

    def set_agent_brains(self, agent_brains: dict):
        """
        Update the agent brains mapping.

        Args:
            agent_brains: Dict mapping agent_id -> LoraBrain
        """
        self.agent_brains = agent_brains

    def process_batch(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """
        Process inference request with per-agent LoRA weights.

        Each agent is processed sequentially with its own LoRA weights loaded
        into the base model before inference.

        Args:
            request: Inference request with prompts and agent IDs

        Returns:
            Inference response with actions and raw outputs
        """
        batch_start = time.time()

        if not request.validate():
            raise ValueError("BatchInferenceRequest has mismatched array lengths")

        actions = []
        raw_outputs = []
        success_flags = []

        total_load_time = 0
        total_generate_time = 0
        total_parse_time = 0

        # Process each agent sequentially with its own LoRA weights
        for agent_id, prompt in zip(request.agent_ids, request.prompts):
            self.inference_count += 1

            try:
                # Load this agent's LoRA weights if available
                load_start = time.time()
                if agent_id in self.agent_brains:
                    brain = self.agent_brains[agent_id]
                    self._load_lora_weights(brain.get_weights())
                load_time = time.time() - load_start
                total_load_time += load_time

                # Generate response
                gen_start = time.time()
                raw_output = self._generate_single(prompt)
                gen_time = time.time() - gen_start
                total_generate_time += gen_time
                raw_outputs.append(raw_output)

                # Parse action
                parse_start = time.time()
                action = self._parse_action(raw_output)
                parse_time = time.time() - parse_start
                total_parse_time += parse_time

                if action:
                    actions.append(action)
                    success_flags.append(True)
                else:
                    actions.append("MOVE_NORTH")  # Default fallback
                    success_flags.append(False)

            except Exception as e:
                print(f"[ERROR] Inference failed for agent {agent_id}: {e}")
                actions.append("MOVE_NORTH")
                raw_outputs.append(f"ERROR: {str(e)}")
                success_flags.append(False)

        total_time = time.time() - batch_start
        num_agents = len(request.agent_ids)
        num_invalid = success_flags.count(False)

        # Return invalid count in the response for caller to report
        # (No printing here - let the simulation handle output)

        return BatchInferenceResponse(
            actions=actions,
            raw_outputs=raw_outputs,
            agent_ids=request.agent_ids,
            success=success_flags
        )

    def _load_lora_weights(self, weights: dict):
        """
        Load LoRA weights into the base model.

        Args:
            weights: Dict of LoRA parameter names -> tensors
        """
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in weights:
                    param.data.copy_(weights[name])

    def _generate_single(self, prompt: str, max_new_tokens: int = 8) -> str:
        """
        Generate response for a single prompt.

        Args:
            prompt: Input prompt (raw content)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Format with chat template - use task-specific system prompt
        messages = [
            {
                "role": "system",
                "content": "You are a survival agent in a grid world. Output ONLY one action: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, or MOVE_WEST. Nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        )

        # Move to same device as model
        device = next(self.base_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with greedy decoding
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens (not the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response

    def _parse_action(self, text: str) -> Optional[str]:
        """
        Parse action from LLM output.

        Args:
            text: Raw LLM output

        Returns:
            Action string (uppercase) or None if not found
        """
        match = self.action_pattern.search(text)
        if match:
            return match.group(1).upper()
        return None
