#!/usr/bin/env python3
"""
Quick Cost Viewer for LIBERO-PRO

A simple tool to quickly view and plot cost data from experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import os
import json
import ast
import numpy as np
import matplotlib.pyplot as plt

def parse_best_traj_cost(cost_field):
    """Parse best_traj_cost which may be a string like '[0.1 0.2]' or a list/array already."""
    if isinstance(cost_field, str):
        # ast.literal_eval will parse "[0.1 0.2]" or "[0.1, 0.2]" into a Python list
        try:
            parsed = ast.literal_eval(cost_field)
        except Exception:
            # fallback: try numpy fromstring (space separated)
            parsed = np.fromstring(cost_field.strip("[]"), sep=' ')
        return np.asarray(parsed, dtype=float)
    else:
        # assume it's list-like already
        return np.asarray(cost_field, dtype=float)


def load_and_plot_costs(cost_file, save_plot=True):
    """Load cost data and create a plot where each future-step cost is expanded into (global_step, cost)."""
    # Load data
    with open(cost_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded data from: {cost_file}")
    print(f"Total episodes: {len(data)}")

    # Create subplots: one for success, one for failure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    success_episodes = []
    failure_episodes = []

    # Separate episodes by success/failure
    for episode_idx, episode in enumerate(data):
        if not episode.get('costs'):
            continue

        # Determine success flag
        succ_flag = episode.get('success')
        if isinstance(succ_flag, str):
            success = succ_flag.lower() == 'true'
        else:
            success = bool(succ_flag)
        
        if success:
            success_episodes.append((episode_idx, episode))
        else:
            failure_episodes.append((episode_idx, episode))

    # Plot success episodes
    for episode_idx, episode in success_episodes:
        flat_steps = []
        flat_costs = []

        for cost_entry in episode['costs']:
            base_step = int(cost_entry['step'])
            best_traj_cost = parse_best_traj_cost(cost_entry['best_traj_cost'])
            # Expand each future cost into its global step (base_step + offset)
            for offset, c in enumerate(best_traj_cost):
                flat_steps.append(base_step + offset)
                flat_costs.append(float(c))

        if not flat_steps:
            continue

        # Convert to numpy arrays and sort by step
        flat_steps = np.array(flat_steps, dtype=int)
        flat_costs = np.array(flat_costs, dtype=float)
        order = np.argsort(flat_steps)
        flat_steps = flat_steps[order]
        flat_costs = flat_costs[order]

        ax1.plot(flat_steps, flat_costs,
                 color='green', alpha=0.8, linewidth=2,
                 label=f"Ep {episode_idx+1} (Success)")

    # Plot failure episodes
    for episode_idx, episode in failure_episodes:
        flat_steps = []
        flat_costs = []

        for cost_entry in episode['costs']:
            base_step = int(cost_entry['step'])
            best_traj_cost = parse_best_traj_cost(cost_entry['best_traj_cost'])
            # Expand each future cost into its global step (base_step + offset)
            for offset, c in enumerate(best_traj_cost):
                flat_steps.append(base_step + offset)
                flat_costs.append(float(c))

        if not flat_steps:
            continue

        # Convert to numpy arrays and sort by step
        flat_steps = np.array(flat_steps, dtype=int)
        flat_costs = np.array(flat_costs, dtype=float)
        order = np.argsort(flat_steps)
        flat_steps = flat_steps[order]
        flat_costs = flat_costs[order]

        ax2.plot(flat_steps, flat_costs,
                 color='red', alpha=0.5, linewidth=1,
                 label=f"Ep {episode_idx+1} (Fail)")

    # Configure success subplot
    ax1.set_xlabel("Global step")
    ax1.set_ylabel("Cost")
    ax1.set_title(f"Success Episodes - Expanded per-step costs ({len(success_episodes)} episodes)")
    ax1.legend()
    ax1.grid(True)

    # Configure failure subplot
    ax2.set_xlabel("Global step")
    ax2.set_ylabel("Cost")
    ax2.set_title(f"Failure Episodes - Expanded per-step costs ({len(failure_episodes)} episodes)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_plot:
        out_path = os.path.splitext(cost_file)[0] + "_separated_costs.png"
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved plot to: {out_path}")
        
    print(f"Success episodes: {len(success_episodes)}")
    print(f"Failure episodes: {len(failure_episodes)}")


def main():
    parser = argparse.ArgumentParser(description='Quick cost viewer for LIBERO-PRO')
    parser.add_argument('--cost_file', default="/hdd/zijianwang/openpi/experiments/cost/LIBERO-PRO-libero_10-20251027_220503/cost_data_task_0.json", help='Path to cost data JSON file')
    parser.add_argument('--no-save', action='store_true', help='Do not save plot to file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cost_file):
        print(f"Error: File {args.cost_file} not found")
        return
    
    load_and_plot_costs(args.cost_file, save_plot=not args.no_save)

if __name__ == "__main__":
    main()
