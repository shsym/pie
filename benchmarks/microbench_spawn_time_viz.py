import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def visualize_spawn_time(log_file_path: str, output_image_path: str):
    """
    Reads spawn time benchmark results from a JSON log file,
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
    data = []
    for log_entry in logs:
        num_instances = log_entry.get('args', {}).get('num_instances')
        mean_latency = log_entry.get('mean_latency')
        stdev_latency = log_entry.get('stdev_latency')

        if all(v is not None for v in [num_instances, mean_latency, stdev_latency]):
            data.append({
                'num_instances': num_instances,
                'mean_latency': mean_latency,
                'stdev_latency': stdev_latency
            })

    if not data:
        print("No valid data found in the log file to plot.")
        return

    df = pd.DataFrame(data)
    # Sort by number of instances to ensure a clean plot
    df = df.sort_values('num_instances').reset_index()


    # --- Visualization ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    plt.errorbar(
        df['num_instances'],
        df['mean_latency'],
        yerr=df['stdev_latency'],
        fmt='-o',  # Format: line with circle markers
        capsize=5,
        elinewidth=2,
        markeredgewidth=2,
        label='Mean Spawn Time'
    )

    # --- Plot Customization ---
    plt.title('Mean Instance Spawn Time vs. Number of Instances', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Instances', fontsize=12)
    plt.ylabel('Mean Spawn Time (μs)', fontsize=12)
    plt.legend(fontsize=10)
    plt.xscale('log', base=2) # Use a log scale for the x-axis
    plt.xticks(df['num_instances'].unique())
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
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
    LOG_FILE = './logs/microbench_spawn_time.json'
    OUTPUT_FILE = './spawn_time_plot.png'
    visualize_spawn_time(LOG_FILE, OUTPUT_FILE)
