"""
Energy Visualisation Module

This module provides visualisation tools for analysing energy metrics and agent behaviour
in the evolutionary simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from matplotlib.ticker import MaxNLocator

class EnergyVisualiser:
    """Class for visualising energy-related metrics from the simulation."""

    def __init__(self, simulation_data: Dict):
        """
        Initialise the EnergyVisualiser with simulation data.

        Args:
            simulation_data: Dictionary containing simulation data including performance history
        """
        self.data = simulation_data
        self.output_dir = "output/energy_plots"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_energy_over_time(self, save_path: str = None) -> None:
        """
        Plot energy metrics over generations.
        In tournament mode, also shows per-tournament energy variance.

        Args:
            save_path: Optional path to save the plot. If None, saves to default location.
        """
        if not self.data.get('performance_history'):
            print("No performance history data available.")
            return

        gens = [m['generation'] for m in self.data['performance_history']]
        max_energy = [m['energy_metrics']['max'] for m in self.data['performance_history']]
        avg_energy = [m['energy_metrics']['avg'] for m in self.data['performance_history']]
        min_energy = [m['energy_metrics']['min'] for m in self.data['performance_history']]

        is_tournament_mode = self.data.get('tournament_mode', False)
        tournament_variance = self.data.get('tournament_energy_variance', [])

        plt.figure(figsize=(12, 6))

        # Plot energy metrics
        plt.plot(gens, max_energy, 'g-', label='Max Energy (Averaged)', linewidth=2)
        plt.plot(gens, avg_energy, 'b-', label='Average Energy', linewidth=2)
        plt.plot(gens, min_energy, 'r-', label='Min Energy (Averaged)', linewidth=2)

        # Add fill between max and min
        plt.fill_between(gens, min_energy, max_energy, color='gray', alpha=0.2, label='Energy Range')

        # In tournament mode, show per-tournament variance for top agents
        if is_tournament_mode and tournament_variance:
            tournament_max_energies = []
            tournament_min_energies = []

            for gen_stats in tournament_variance:
                if gen_stats:
                    # Get the top 3 agents' tournament energies
                    top_agents = sorted(gen_stats, key=lambda x: x['mean_energy'], reverse=True)[:3]

                    # For each agent, get their max and min energy across tournaments
                    agent_tournament_maxes = [max(agent['tournament_energies']) for agent in top_agents if agent['tournament_energies']]
                    agent_tournament_mins = [min(agent['tournament_energies']) for agent in top_agents if agent['tournament_energies']]

                    if agent_tournament_maxes and agent_tournament_mins:
                        tournament_max_energies.append(max(agent_tournament_maxes))
                        tournament_min_energies.append(min(agent_tournament_mins))
                    else:
                        tournament_max_energies.append(max_energy[len(tournament_max_energies)])
                        tournament_min_energies.append(min_energy[len(tournament_min_energies)])
                else:
                    # No tournament data for this generation
                    tournament_max_energies.append(max_energy[len(tournament_max_energies)])
                    tournament_min_energies.append(min_energy[len(tournament_min_energies)])

            # Plot tournament variance as dotted lines
            if len(tournament_max_energies) == len(gens):
                plt.plot(gens, tournament_max_energies, 'g:', label='Tournament Max', alpha=0.6, linewidth=1)
                plt.plot(gens, tournament_min_energies, 'r:', label='Tournament Min', alpha=0.6, linewidth=1)
                plt.fill_between(gens, tournament_min_energies, tournament_max_energies,
                               color='orange', alpha=0.1, label='Tournament Variance')

        title = 'Energy Metrics Over Generations'
        if is_tournament_mode:
            title += ' (Tournament Mode - Showing Per-Tournament Variance)'
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Energy')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)

        # Ensure x-axis shows integer generation numbers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save or show the plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'energy_over_time.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Energy over time plot saved to {save_path}")
    
    def plot_population_health(self, save_path: str = None) -> None:
        """
        Plot population health metrics (alive/dead counts) over generations.
        
        Args:
            save_path: Optional path to save the plot.
        """
        if not self.data.get('performance_history'):
            print("No performance history data available.")
            return
            
        gens = [m['generation'] for m in self.data['performance_history']]
        alive = [m['energy_metrics']['alive_count'] for m in self.data['performance_history']]
        dead = [m['energy_metrics']['dead_count'] for m in self.data['performance_history']]
        
        plt.figure(figsize=(12, 6))
        
        # Plot stacked bar chart of alive/dead agents
        plt.bar(gens, alive, label='Alive Agents', color='green', alpha=0.7)
        plt.bar(gens, dead, bottom=alive, label='Dead Agents', color='red', alpha=0.7)
        
        plt.title('Population Health Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Number of Agents')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Ensure x-axis shows integer generation numbers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'population_health.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Population health plot saved to {save_path}")

    def generate_all_plots(self) -> None:
        """Generate all available plots and save them to the output directory."""
        self.plot_energy_over_time()
        self.plot_population_health()
        print(f"All energy visualisations saved to {self.output_dir}")


def visualise_energy_metrics(simulation_data: Dict) -> None:
    """
    Convenience function to generate all energy visualisations.

    Args:
        simulation_data: Dictionary containing simulation data
    """
    visualiser = EnergyVisualiser(simulation_data)
    visualiser.generate_all_plots()


if __name__ == "__main__":
    # Example usage
    import json

    # Load simulation data
    with open("output/simulation_data.json", "r") as f:
        data = json.load(f)

    # Create visualisations
    visualiser = EnergyVisualiser(data)
    visualiser.generate_all_plots()
