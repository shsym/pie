import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def categorize_run(row):
    """Categorizes a run based on its boolean flags."""
    use_prefix_cache = row.get('use_prefix_cache', False)
    concurrent_calls = row.get('concurrent_calls', False)
    drop_tool_cache = row.get('drop_tool_cache', False)

    if use_prefix_cache and concurrent_calls and drop_tool_cache:
        return "Case 4: Prefix Cache + Concurrent Calls + Drop Tool Cache"
    elif use_prefix_cache and concurrent_calls:
        return "Case 3: Prefix Cache + Concurrent Calls"
    elif use_prefix_cache:
        return "Case 2: Prefix Cache"
    else:
        return "Case 1: Baseline (No optimization)"

def visualize_logs(log_file_path: str):
    """
    Reads a JSON log file, processes the data, and generates a plot
    of throughput vs. number of instances for different experiment cases.
    """
    log_path = Path(log_file_path)
    if not log_path.is_file():
        print(f"Error: Log file not found at '{log_file_path}'")
        return

    # Read and parse the JSON file
    with open(log_path, 'r', encoding='utf-8') as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{log_file_path}'.")
            return

    if not logs:
        print("Log file is empty. Nothing to visualize.")
        return

    # Convert to pandas DataFrame
    df = pd.DataFrame(logs)

    # The 'args' column is a dict, so we expand it into its own columns
    # and combine it with the main DataFrame.
    args_df = pd.json_normalize(df['args'])
    df = pd.concat([df.drop(columns=['args']), args_df], axis=1)

    # Add a 'Case' column for categorization
    df['Case'] = df.apply(categorize_run, axis=1)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get unique cases and assign colors/markers for plotting
    cases = sorted(df['Case'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(cases)))

    for i, case in enumerate(cases):
        # Filter data for the current case and sort by num_instances
        case_df = df[df['Case'] == case].sort_values('num_instances')
        if not case_df.empty:
            ax.plot(
                case_df['num_instances'],
                case_df['throughput'],
                marker='o',
                linestyle='-',
                label=case,
                color=colors[i]
            )

    # --- Formatting the plot ---
    ax.set_title('Benchmark Throughput vs. Number of Instances', fontsize=18, pad=20)
    ax.set_xlabel('Number of Concurrent Instances', fontsize=14)
    ax.set_ylabel('Throughput (requests/second)', fontsize=14)
    ax.legend(title='Experiment Case', fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-axis to log scale for better visualization of instance counts
    ax.set_xscale('log')
    unique_instances = sorted(df['num_instances'].unique())
    ax.set_xticks(unique_instances)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot to a file
    output_filename = 'benchmark_throughput.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize benchmark logs from a JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the JSON log file.",
        default="./logs/test_4_agent_case_study_pie.json",
        nargs='?'  # Makes the argument optional with a default value
    )
    args = parser.parse_args()
    visualize_logs(args.log_file)
