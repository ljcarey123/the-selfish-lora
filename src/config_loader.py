import json
import os
from dotenv import load_dotenv

# Build the path to the project root from this file's location (src/ -> root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from the .env file in the project root
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

# Get the path to the prompts file from the environment variable
_prompts_file_path = os.getenv('PROMPTS_FILE_PATH')

_config = {}
# Construct the full path to the prompts.json file
if _prompts_file_path:
    # Assume the path in .env is relative to the project root
    config_json_path = os.path.join(project_root, _prompts_file_path)
    
    if os.path.exists(config_json_path):
        with open(config_json_path, 'r') as f:
            _config = json.load(f)
    else:
        print(f"Warning: Prompts config file not found at '{config_json_path}'. Using empty defaults.")
else:
    print("Warning: PROMPTS_FILE_PATH not set in .env file. Using empty defaults.")

# Grid-related constants
GRID_SIZE = 5  # Perception grid size (5x5 window)
AGENT_POS = (2, 2)  # Agent always at center of perception

# Expose the configuration values as module-level constants
SYSTEM_PROMPT = _config.get('SYSTEM_PROMPT', '')
VALID_ACTIONS = _config.get('VALID_ACTIONS', [])
ENERGY_LEVELS = _config.get('ENERGY_LEVELS', {})
GRID_SYMBOLS = _config.get('GRID_SYMBOLS', {})
