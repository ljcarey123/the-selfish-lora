# Phase 1: Minimum Viable Product (MVP) - "The Core Loop"

## 🎯 Objective
The primary goal of this phase is to build and validate the fundamental evolutionary mechanism. We need to prove that the core loop of selection, reproduction, and mutation can successfully breed agents with increasing fitness in a controlled and repeatable environment.

---

## 🔧 Core Components

### 1. Environment
* **Structure:** A fixed, static 50x50 grid.
* **Resources:** A single, unchanging layout of resources.
* **Purpose:** This controlled environment acts as a clear benchmark to validate that the evolutionary process is working and to establish a performance baseline.

### 2. Agent
* **Actions:** A minimal set of actions: `move` (North, South, East, West) and `gather`.
* **Perception:** Agents have a complete view of the entire grid state.
* **Decision Making (Technical Approach):**
    * **Engine:** Each agent's "brain" is a base LLM (e.g., Gemma) modified by its own unique LoRA adapter.
    * **Input (Prompt):** For each turn, the agent's entire perception of the grid is serialized into a text string (e.g., `GRID_STATE: [0,0,2,0,...] AGENT_POS: (x,y)`) and fed into the LLM.
    * **Output (Action):** The LLM is prompted to return a single, parsable action command from the allowed list (e.g., "MOVE N", "GATHER"). The simulation code executes this command.

### 3. Evolutionary Loop
* **Initialization (Generation 0):** The initial population is created with a set of LoRA adapters where the `A` matrices are randomly initialized and the `B` matrices are set to zero. This creates a diverse, unbiased starting gene pool.
* **Fitness Function:** A simple, direct metric: `fitness_score = number_of_resources_collected`.
* **Selection:** Truncation selection. At the end of a generation, the lowest-performing agents are culled, and the top performers are selected for reproduction.
* **Reproduction (Heredity):** A new "child" **LoRA adapter** is created by averaging the **weight matrices** of two "parent" adapters (`child_weights = 0.5 * parent1_weights + 0.5 * parent2_weights`).
* **Mutation (Variation):** A small amount of random noise is added to the child **adapter's weight matrices** to introduce new traits.

---

## ✅ Success Criterion
* **Primary Metric:** Observe a clear, statistically significant upward trend in the average population fitness score over at least 50 generations. This will confirm that the core evolutionary algorithm is functioning as intended.