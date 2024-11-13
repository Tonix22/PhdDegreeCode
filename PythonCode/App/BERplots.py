import pandas as pd
import matplotlib.pyplot as plt
import os


# Add sys.path for accessing the required modules

def plot_csv_files(csv_files, output_filename="output_plot.png"):
    """
    Reads multiple CSV files, plots their data with different styles and colors, 
    and saves the plot.

    Parameters:
        csv_files (list of str): List of CSV file paths to be read and plotted.
        output_filename (str): The output file name for the plot (default: 'output_plot.png').
    """
    # Predefined styles and colors for each plot
    styles = ['-', '--', '-.', ':']  # Different line styles
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Different colors
    
    plt.figure(figsize=(10, 6))

    for i, file in enumerate(csv_files):
        try:
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
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)

    # Save the plot to the current path with the specified output filename
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved as {output_filename}")

# Example usage

csv_files = ["ber_snr_data_clean.csv",  "ber_snr_data_NN.csv"]
output_filename = "combined_plot.png"
plot_csv_files(csv_files, output_filename)
