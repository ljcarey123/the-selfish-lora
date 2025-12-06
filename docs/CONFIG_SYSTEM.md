# Game Mechanics Configuration System

## Overview

All game mechanics are now centralized in [config/game_mechanics.json](../config/game_mechanics.json). This ensures consistency between simulation and visualization, and makes it easy to experiment with different game parameters.

## Changes Made

### 1. Created Configuration File

**File**: [config/game_mechanics.json](../config/game_mechanics.json)

Contains all game mechanics parameters:
- Initial energy: 100
- Metabolism cost: 2 per turn
- Food reward: 30 energy
- Combat attacker gain: 10 energy (stolen from victim)
- Combat victim loss: 10 energy
- Death threshold: 0 energy

### 2. Updated Environment

**File**: [src/Environment/environment_vectorized.py](../src/Environment/environment_vectorized.py)

Changes:
- Loads `game_mechanics.json` in `__init__()` (line 60-62)
- Uses config for initial energy in `register_agents()` (line 173-174)
- Uses config for combat mechanics in `execute_actions()` (line 470-472)
- Uses config for food reward (line 495-498)
- Uses config for metabolism cost in `apply_metabolism()` (line 537-539)
- Uses config for combat damage in `apply_combat_damage()` (line 526-527)
- Uses config for food energy tracking (line 161-162)

### 3. Updated Visualizer

**File**: [src/visualisation/visualiser.py](../src/visualisation/visualiser.py)

Changes:
- Loads `game_mechanics.json` in `__init__()` (line 38-41)
- Uses config in `_create_action_timeline()` for accurate energy tracking (line 631-636)
- **Fixed combat visualization bug**: Attackers now correctly show energy GAIN (+10) instead of just metabolism cost (-2)
- Timeline now shows:
  - **FOOD_COLLECTED**: +30 energy, -2 metabolism = +28 net
  - **COMBAT_ATTACK**: +10 energy stolen, -2 metabolism = +8 net ✅ FIXED
  - **COMBAT_VICTIM**: -10 damage, -2 metabolism = -12 net
  - **MOVE**: -2 metabolism only

### 4. Documentation

Created [config/README.md](../config/README.md) explaining the configuration system and how to modify parameters.

## Key Benefits

1. **Single Source of Truth**: Game mechanics defined once, used everywhere
2. **Easy Experimentation**: Change one config file to test different game parameters
3. **Consistency**: Simulation and visualization always use the same values
4. **Bug Fix**: Visualization timeline now correctly shows attackers gaining energy

## Example: Making Combat More Aggressive

Edit `config/game_mechanics.json`:

```json
{
  "energy": {
    "combat_attacker_gain": 15,
    "combat_victim_loss": 15
  }
}
```

Both simulation and visualization will automatically use the new values.

## Combat Mechanics Clarification

**Before this fix**, the visualization showed attackers losing energy when attacking, which was incorrect.

**After this fix**, the energy flow is now accurate:
- Attacker: Gains +10 energy (steals from victim), pays -2 metabolism = **+8 net gain**
- Victim: Loses -10 energy (stolen by attacker), pays -2 metabolism = **-12 net loss**

This encourages aggressive behavior - attacking is profitable for the attacker!
