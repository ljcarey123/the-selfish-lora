# Metrics and Visualization Plan

This document outlines a comprehensive plan for collecting and visualizing data from the simulation. The goal is to create a powerful dashboard that allows us to understand not just *what* is happening, but *how* and *why* it is happening.

---

## 1. Population Performance Metrics

*   **Objective:** To measure the overall success of the population and its trend over time.

*   **Metrics to Collect (per generation):**
    *   Average Fitness
    *   Maximum Fitness
    *   Minimum Fitness
    *   Median Fitness
    *   Standard Deviation of Fitness

*   **Implementation Plan:**
    1.  In the `Simulation` class, create a list `self.performance_history = []`.
    2.  At the end of each generation (after `run_generation`), calculate the above metrics from the `fitness_score` of all agents in `self.population`.
    3.  Append a dictionary containing these metrics to `self.performance_history`.

*   **Visualization Plan:**
    *   **Chart Type:** Line Chart.
    *   **Title:** "Population Fitness Over Generations".
    *   **X-Axis:** Generation Number.
    *   **Y-Axis:** Fitness Score.
    *   **Content:** Plot `Max Fitness`, `Average Fitness`, and `Min Fitness` as separate lines. A shaded area between the max and min lines can effectively visualize the performance spread.

---

## 2. Genetic Metrics

*   **Objective:** To measure the convergence or divergence of the agent gene pool.

*   **Metrics to Collect (per generation):**
    *   Genetic Diversity (average Euclidean distance between the weight matrices of all agent pairs).

*   **Implementation Plan:**
    1.  In the `Simulation` class, create a list `self.genetic_history = []`.
    2.  Create a new method `_calculate_genetic_diversity()`. This method will iterate through a sample of agent pairs, calculate the distance between their `brain.weights`, and return the average.
    3.  At the end of each generation, call this method and append the result to `self.genetic_history`.

*   **Visualization Plan:**
    *   **Chart Type:** Line Chart.
    *   **Title:** "Genetic Diversity Over Generations".
    *   **X-Axis:** Generation Number.
    *   **Y-Axis:** Average Genetic Distance.

---

## 3. Behavioral Metrics

*   **Objective:** To understand how the collective strategy of the population changes over time.

*   **Metrics to Collect (per generation):**
    *   Action Distribution (a count of each action type: `MOVE_N`, `MOVE_S`, `GATHER`, etc.).

*   **Implementation Plan:**
    1.  In the `Agent` class, add a list `self.action_history = []`.
    2.  In `_execute_action`, append the chosen `action` string to this list.
    3.  In the `Simulation` class, create a list `self.behavioral_history = []`.
    4.  At the end of a generation, iterate through all agents, aggregate the counts of each action from their `action_history`, and append the totals to `self.behavioral_history`.

*   **Visualization Plan:**
    *   **Chart Type:** Stacked Area Chart.
    *   **Title:** "Action Distribution Over Generations".
    *   **X-Axis:** Generation Number.
    *   **Y-Axis:** Percentage of Actions.
    *   **Content:** Each action type will be a colored band, showing its proportional representation over time.

---

## 4. Qualitative Behavioral Analysis

*   **Objective:** To visually inspect and understand the specific strategies employed by individual agents.

*   **Metrics to Collect:**
    *   **Champion's Path (per generation):** The full sequence of `(x, y)` coordinates for the single best agent.
    *   **Final Population Paths (final generation only):** The full sequence of coordinates for *all* agents in the last generation.

*   **Implementation Plan:**
    1.  In the `Agent` class, add a list `self.path_history = []` and record the agent's position after every turn.
    2.  In the `Simulation` class, create `self.champion_history = {}` and `self.final_generation_history = []`.
    3.  At the end of each generation, find the champion and store its `path_history`.
    4.  If it is the final generation, store the `path_history` for all agents.

*   **Visualization Plan:**
    *   **Tool:** A custom visualization script using a library like `matplotlib` or `pygame`.
    *   **Functionality:**
        1.  **Champion Replay:** A dropdown menu to select a generation number. Upon selection, it renders the environment grid and animates the champion's path for that generation, showing its movement turn by turn.
        2.  **Final Generation Replay:** An animation that shows all agents of the final generation moving simultaneously on the grid, providing a view of the final, evolved ecosystem in action.
