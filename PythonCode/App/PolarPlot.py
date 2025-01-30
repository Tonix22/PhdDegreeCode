import numpy as np
import matplotlib.pyplot as plt

def plot_two_polar_arrays(array1, array2, start=0, end=None):
    """
    Plots two polar plots (in a side-by-side figure) from two complex arrays.
    
    Parameters
    ----------
    array1 : numpy.ndarray
        First input complex array.
    array2 : numpy.ndarray
        Second input complex array.
    start : int, optional
        Start index (inclusive) for slicing the arrays. Default is 0.
    end : int, optional
        End index (exclusive) for slicing the arrays. If None, slices to the end.
    """
    
    # Handle default value for end
    if end is None:
        end = len(array1)  # or use min(len(array1), len(array2)) if they differ
    
    # Slice the arrays
    arr1 = array1[start:end]
    arr2 = array2[start:end]
    
    # Convert complex arrays to polar coordinates
    # r = magnitude, theta = angle
    r1 = np.abs(arr1)
    theta1 = np.angle(arr1)
    
    r2 = np.abs(arr2)
    theta2 = np.angle(arr2)
    
    # Prepare colors for each data point. 
    # One approach is to use a colormap with as many unique values as data points.
    colors1 = np.linspace(0, 1, len(r1))
    colors2 = np.linspace(0, 1, len(r2))
    
    # Create figure and two polar subplots
    fig = plt.figure(figsize=(10, 5))
    
    # First polar plot
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    scatter1 = ax1.scatter(theta1, r1, c=colors1, cmap='rainbow', alpha=0.75)
    ax1.set_title("Polar Plot 1")
    
    # Second polar plot
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    scatter2 = ax2.scatter(theta2, r2, c=colors2, cmap='rainbow', alpha=0.75)
    ax2.set_title("Polar Plot 2")
    
    # Optionally add colorbars (one for each subplot)
    # They will reflect the index distribution of the plotted points.
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.1)
    cbar1.set_label('Index (normalized) of array1')
    cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.1)
    cbar2.set_label('Index (normalized) of array2')
    
    plt.tight_layout()
    plt.show()

"""
# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    t = np.linspace(0, 6*np.pi, 100)
    data1 = np.exp(1j * t)  # Complex exponential
    data2 = np.exp(1j * 2 * t)  # Different frequency
    
    # Plot only a subset of the data
    plot_two_polar_arrays(data1, data2, start=10, end=12)
"""