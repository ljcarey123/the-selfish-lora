# Phase 2: Core Extensions - "Forcing Generalization"

## 🎯 Objective
Once the MVP is validated, the goal is to introduce mechanics that prevent agents from simply "memorizing" a single solution. This phase forces the evolution of more robust, intelligent, and adaptive behaviors that can generalize to changing conditions. These extensions should be implemented and tested one at a time.

---

## 🔧 Extensions (Implement Sequentially)

### 1. Extension 2A: Dynamic Environment
* **Change:** The environment's resource layout (positions and scarcity) will be randomly generated at the start of each new generation.
* **Purpose:** This is the most critical extension. It forces agents to evolve general strategies for finding resources rather than memorizing the layout of a single, static map.
* **Technical Approach:**
    * **Prompt Engineering:** The entire grid state will be serialized into a text format (e.g., a string of numbers and characters) and fed into the LLM's prompt for each turn.
    * **LoRA's Role:** The LoRA adapter's function is to fine-tune the base LLM's general spatial reasoning abilities. It must evolve to interpret the serialized grid data and output a valid action (`move` or `gather`), effectively becoming a specialized policy network for this environment. The dynamic map prevents the LoRA from simply memorizing coordinate-to-action mappings.

### 2. Extension 2B: Seeded LoRA Population
* **Change:** Instead of starting with random LoRAs, initialize the Generation 0 population from a common "seed" LoRA.
* **Hypothesis:** This will lead to more stable and rapid fitness improvement in the dynamic environment by providing a more coherent starting point for evolution.
* **Technical Approach:**
    * **Seed Creation:** Before the simulation begins, a single "base" LoRA will be fine-tuned for a very small number of steps on a simple, curated dataset (e.g., 10-20 examples of an agent moving towards an obviously placed resource).
    * **Population Generation:** This single, lightly-trained "seed" LoRA is then copied for every agent in Generation 0. A small, unique mutation (random noise) is applied to each copy to create the initial diversity in the population's gene pool.

### 3. Extension 2C: Simple Communication
* **Change:** Introduce a shared "blackboard" system where agents can write a single character from a limited set (e.g., 'R' for resource).
* **Hypothesis:** A rudimentary signaling protocol will emerge, where symbols become correlated with environmental states or agent behaviors, potentially leading to cooperative strategies.
* **Technical Approach:**
    * **Expanded Action Space:** The LLM's required output format is expanded. It must now choose between `move`, `gather`, and `write(character)`. The simulation code will need to parse this more complex output.
    * **Modified Prompt:** The blackboard's state will be added to the LLM's input prompt alongside the grid state. For example: `GRID: [grid_data] BLACKBOARD: [blackboard_data] ACTION: ?`.
    * **Evolved Interpretation:** The evolutionary pressure will be on the LoRA to not only parse this more complex prompt but to correlate the blackboard data with the grid data. Success will depend on agents evolving the ability to both produce meaningful signals and interpret the signals of others to improve their resource gathering.

---

## ✅ Success Criteria
* **Dynamic Environment:** Agents must maintain a positive fitness trend over generations, even with a new map each time.
* **Seeded LoRA:** Observe a faster or more stable convergence of the fitness score compared to the randomly initialized population.
* **Communication:** Discover a statistically significant correlation between the symbols written to the blackboard and the state of the game (e.g., the proximity of resources to the signaling agent).