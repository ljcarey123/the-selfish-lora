"""
Prepares multiple parent models for multi-parent evolution initialization.

This script copies the top N fine-tuned runs (by evolution score) to src/weights/
for use in evolutionary simulation with increased genetic diversity.

Usage:
    python prepare_parents.py --top 3
    python prepare_parents.py --runs run-0 run-3 run-7
"""

import os
import shutil
import json
import argparse


def get_run_metrics(output_dir):
    """Extract metrics from EVALUATION_REPORT_FINAL.txt."""
    eval_report_path = os.path.join(output_dir, "EVALUATION_REPORT_FINAL.txt")
    
    if not os.path.exists(eval_report_path):
        print(f"Error: {eval_report_path} not found!")
        print("Please run evaluate_model.py first to generate evaluation metrics.")
        return []
    
    runs = []
    
    # Parse the evaluation report
    with open(eval_report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the performance table
    in_table = False
    skip_header = 0
    for line in lines:
        if "Fine-Tuned Runs Performance" in line:
            in_table = True
            skip_header = 3  # Skip the next 3 lines (====, header row, ----)
            continue
        
        if in_table:
            if skip_header > 0:
                skip_header -= 1
                continue
            
            # Check if we've reached the end of the table
            if line.strip().startswith("="):
                break
            
            # Parse data line: run-X    evol_score  valid%  div%  bal%  acc%
            if line.strip().startswith("run-"):
                parts = line.split()
                if len(parts) >= 6:
                    run_name = parts[0]
                    evolution_score = float(parts[1])
                    valid_rate = float(parts[2])
                    diversity = float(parts[3])
                    balance = float(parts[4])
                    accuracy = float(parts[5])
                    
                    # Find checkpoint path
                    run_path = os.path.join(output_dir, run_name)
                    checkpoint_path = None
                    if os.path.exists(run_path):
                        for subitem in os.listdir(run_path):
                            if subitem.startswith("checkpoint-"):
                                checkpoint_path = os.path.join(run_path, subitem)
                                break
                    
                    if checkpoint_path:
                        runs.append({
                            'name': run_name,
                            'path': checkpoint_path,
                            'evolution_score': evolution_score,
                            'valid_rate': valid_rate,
                            'diversity': diversity,
                            'balance': balance,
                            'accuracy': accuracy,
                        })
    
    return runs


def prepare_parents(output_dir, weights_dir, run_names=None, top_n=None):
    """Copy selected runs to weights directory."""
    
    # Clear existing weights directory
    if os.path.exists(weights_dir):
        print(f"Clearing existing weights directory: {weights_dir}")
        shutil.rmtree(weights_dir)
    os.makedirs(weights_dir)
    
    # Get all runs
    all_runs = get_run_metrics(output_dir)
    
    if not all_runs:
        print("No runs found in output directory!")
        return
    
    # Select runs
    if run_names:
        # Use specific run names
        selected_runs = [r for r in all_runs if r['name'] in run_names]
        if len(selected_runs) != len(run_names):
            found = [r['name'] for r in selected_runs]
            missing = set(run_names) - set(found)
            print(f"Warning: Could not find runs: {missing}")
    elif top_n:
        # Select top N by evolution score (already calculated in evaluation)
        selected_runs = sorted(all_runs, key=lambda x: x['evolution_score'], reverse=True)[:top_n]
    else:
        # Use all runs
        selected_runs = all_runs
    
    # Copy selected runs
    print(f"\nPreparing {len(selected_runs)} parent models (sorted by evolution score):")
    for run in selected_runs:
        src = run['path']
        dst = os.path.join(weights_dir, run['name'])
        
        print(f"\n  {run['name']}:")
        print(f"    Evolution Score:  {run['evolution_score']:.1f}%")
        print(f"    Valid Rate:       {run['valid_rate']*100:.1f}%")
        print(f"    Diversity:        {run['diversity']*100:.1f}%")
        print(f"    Balance:          {run['balance']*100:.1f}%")
        print(f"    Accuracy:         {run['accuracy']:.1f}%")
        
        # Copy adapter files
        shutil.copytree(src, dst)
        print(f"    Copied to: {dst}")
    
    print(f"\n✅ Successfully prepared {len(selected_runs)} parent models in {weights_dir}")
    print(f"\nTo use in simulation:")
    print(f"  python main.py --population 20 --generations 20")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare parent models for multi-parent evolution")
    parser.add_argument("--top", type=int, help="Select top N runs by evolution score")
    parser.add_argument("--runs", nargs="+", help="Specific run names to use (e.g., run-0 run-3)")
    parser.add_argument("--output-dir", default="output", help="Fine-tuning output directory")
    parser.add_argument("--weights-dir", default="../src/weights", help="Target weights directory")
    
    args = parser.parse_args()
    
    if not args.top and not args.runs:
        print("Error: Must specify either --top N or --runs run-0 run-1 ...")
        parser.print_help()
        exit(1)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    weights_dir = os.path.join(script_dir, args.weights_dir)
    
    prepare_parents(output_dir, weights_dir, args.runs, args.top)
