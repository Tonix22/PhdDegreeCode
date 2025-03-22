import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_config(json_path):
    """Load configuration from JSON file."""
    with open(json_path, 'r') as file:
        config = json.load(file)
    return config

def plot_csv_files(csv_files, output_filename="output_plot.png"):
    """
    Reads multiple CSV files, plots their data with different styles and colors, 
    and saves the plot.

    Parameters:
        csv_files (list of str): List of CSV file paths to be read and plotted.
        output_filename (str): The output file name for the plot (default: 'output_plot.png').
    """
    # Predefined styles and colors for each plot
    styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(10, 6))

    for i, file in enumerate(csv_files):
        try:
            if not os.path.isfile(file):
                print(f"Warning: File not found {file}")
                continue
            
            # Read the CSV file
            data = pd.read_csv(file)
            
            # Select a style and color based on the file index
            style = styles[i % len(styles)]
            color = colors[i % len(colors)]
            
            # Assuming the CSV has two columns: one for x and one for y
            plt.semilogy(data.iloc[:, 0], data.iloc[:, 1], style, color=color, label=os.path.basename(file))
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    plt.title("Combined Plot of CSV Data")
    plt.ylabel("BER")
    plt.xlabel("SNR dB")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved as {output_filename}")

def main(json_path):
    """Main function to load configuration and generate plots."""
    config = load_config(json_path)
    
    csv_files = config.get("csv_files", [])
    output_filename = config.get("output_filename", "output_plot.png")

    if not csv_files:
        print("Error: No CSV files provided in the JSON configuration.")
        return
    
    plot_csv_files(csv_files, output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CSV data from a JSON configuration file.")
    parser.add_argument("config_json", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    main(args.config_json)
