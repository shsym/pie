import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def visualize_execution_latency(log_file_path: str, output_image_path: str):
    """
    Reads execution latency benchmark results from a JSON log file,
    processes the data, and generates a plot.

    Args:
        log_file_path (str): The path to the JSON log file.
        output_image_path (str): The path to save the output plot image.
    """
    log_file = Path(log_file_path)
    if not log_file.is_file():
        print(f"Error: Log file not found at '{log_file_path}'")
        return

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{log_file_path}'.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- Data Processing ---
    # Extract relevant data and create a pandas DataFrame
    data = []
    for log_entry in logs:
        num_instances = log_entry.get('args', {}).get('num_instances')
        layer = log_entry.get('args', {}).get('layer')
        mean_latency = log_entry.get('mean_latency')
        stdev_latency = log_entry.get('stdev_latency')

        if all(v is not None for v in [num_instances, layer, mean_latency, stdev_latency]):
            data.append({
                'num_instances': num_instances,
                'layer': layer,
                'mean_latency': mean_latency,
                'stdev_latency': stdev_latency
            })

    if not data:
        print("No valid data found in the log file to plot.")
        return

    df = pd.DataFrame(data)

    # --- Visualization ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Create a line plot for each layer
    for layer_name, group in df.groupby('layer'):
        # Sort values by number of instances to ensure the line plot connects points correctly
        group = group.sort_values('num_instances')
        plt.errorbar(
            group['num_instances'],
            group['mean_latency'],
            yerr=group['stdev_latency'],
            label=f'{layer_name.capitalize()} Layer',
            fmt='-o', # Format: line with circle markers
            capsize=5, # Error bar cap size
            elinewidth=2,
            markeredgewidth=2
        )

    # --- Plot Customization ---
    plt.title('Mean Execution Latency vs. Number of Instances', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Instances', fontsize=12)
    plt.ylabel('Mean Latency (μs)', fontsize=12)
    plt.legend(title='Layer Type', fontsize=10)
    plt.xscale('log', base=2) # Use a log scale for the x-axis for better visualization
    plt.xticks(df['num_instances'].unique()) # Ensure all instance counts are shown as ticks
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Format x-ticks as numbers
    plt.grid(True, which="both", ls="--")

    # --- Save and Show Plot ---
    try:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{output_image_path}'")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()

if __name__ == '__main__':
    # Make sure the log directory and file name match your setup
    LOG_FILE = './logs/microbench_execution_latency.json'
    OUTPUT_FILE = './execution_latency_plot.png'
    visualize_execution_latency(LOG_FILE, OUTPUT_FILE)
