# Configuration Files

This directory contains configuration files for the Selfish LoRA simulation.

## Files

### game_mechanics.json

Contains all game mechanics parameters that affect gameplay:

- **energy**:
  - `initial`: Starting energy for each agent (default: 100)
  - `metabolism_cost`: Energy cost per turn for all actions (default: 2)
  - `food_reward`: Energy gained from collecting food (default: 30)
  - `combat_attacker_gain`: Energy stolen by attacker in combat (default: 10)
  - `combat_victim_loss`: Energy lost by victim in combat (default: 10)
  - `death_threshold`: Energy level at which agents die (default: 0)

- **combat**:
  - `enabled`: Whether combat is enabled (default: true)
  - `energy_transfer`: Whether attackers steal energy from victims (default: true)

- **food**:
  - `regrowth_rate`: Probability of food regrowth per turn (default: 0.05)

### prompts.json

Contains prompt templates and system messages for the LLM agents.

## Usage

These configuration files are automatically loaded by:
- `environment_vectorized.py` - Reads game mechanics for simulation rules
- `visualiser.py` - Reads game mechanics for accurate energy timeline visualization

All game mechanics are centralized in `game_mechanics.json` to ensure consistency between simulation and visualization.

## Example: Changing Combat Mechanics

To make combat more aggressive, edit `game_mechanics.json`:

```json
{
  "energy": {
    "combat_attacker_gain": 15,
    "combat_victim_loss": 15
  }
}
```

This will make attackers steal 15 energy instead of 10, and victims will lose 15 energy.
