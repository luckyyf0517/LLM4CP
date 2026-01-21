"""
@Project : LLM4CP
@File    : plotting.py
@IDE     : PyCharm
@Author  : XvanyvLiu
@mail    : xvanyvliu@gmail.com
@Date    : 2024/1/21

Visualization utilities for test results.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Set plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3


def plot_nmse(json_path="./outputs/results_fdd_full.json", save_path="./outputs/nmse_vs_velocity.png"):
    """
    Plot NMSE vs Velocity from test results JSON file.

    Args:
        json_path: Path to JSON file containing test results.
        save_path: Path to save the plot.
    """
    # Load results from JSON
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Extract NMSE data
    nmse_data = []
    model_names = []
    for model in results['models']:
        if model['nmse_per_velocity']:
            nmse_data.append(model['nmse_per_velocity'])
            model_names.append(model['name'])

    # Velocity mapping (speed 0-9 -> 10-100 km/h)
    velocities = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

    # Plot each model
    for i, (nmse_values, name) in enumerate(zip(nmse_data, model_names)):
        if len(nmse_values) == len(velocities):
            ax.plot(velocities, nmse_values,
                   marker=markers[i % len(markers)],
                   markersize=8,
                   linewidth=2,
                   linestyle=linestyles[i % len(linestyles)],
                   color=colors[i % len(colors)],
                   label=name)

    # Set axis labels and title
    ax.set_xlabel('Velocity (km/h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('NMSE', fontsize=14, fontweight='bold')
    ax.set_title('NMSE vs Velocity', fontsize=16, fontweight='bold')

    # Set y-axis to linear scale
    ax.set_ylim([0.0, 1.0])

    # Set x-axis limits
    ax.set_xlim([0, 110])

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=1)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Tight layout
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")


def save_averages(json_path="./outputs/results_fdd_full.json", output_path="./outputs/averages.json"):
    """
    Save average SE and BER results to a separate JSON file.

    Args:
        json_path: Path to JSON file containing test results.
        output_path: Path to save the average results JSON.
    """
    # Load results from JSON
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Extract average results
    averages = {
        "timestamp": results.get("timestamp", ""),
        "models": []
    }

    for model in results['models']:
        avg_data = {
            "name": model['name'],
            "se_avg": model.get('se_avg', 0.0),
            "ber_avg": model.get('ber_avg', 0.0)
        }
        averages['models'].append(avg_data)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(averages, f, indent=4)

    print(f"Average results saved to: {output_path}")
    print("\nAverage SE and BER:")
    print("-" * 60)
    for model in averages['models']:
        print(f"{model['name']:12s} | SE: {model['se_avg']:8.4f} bit/(s·Hz) | BER: {model['ber_avg']:.6f}")
    print("-" * 60)
