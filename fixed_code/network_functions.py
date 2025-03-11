import matplotlib.pyplot as plt
import numpy as np

def plot(X, Theta, model_type, R):
    """
    Plot the data samples and network approximation.
    
    Args:
        X: Dataset containing (x, t) pairs
        Theta: Network parameters (W1, w2)
        model_type: Type of network (1, 2, or 3)
        R: Range [min, max] for x-axis
    """
    # Extract x and t values from dataset
    x_data = [x[0][1] for x in X]  # Extract x from [1, x]
    t_data = [t for _, t in X]     # Extract target values
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot data samples
    plt.scatter(x_data, t_data, color='red', marker='x', alpha=0.5, label="Data")
    
    # Define equidistant points from min (R[0]) to max (R[1]) to evaluate the network
    x = np.linspace(R[0], R[1], 200)
    
    # Compute the network outputs for these values
    y = []
    for x_val in x:
        # Create input vector [1, x]
        x_vec = np.array([1, x_val])
        # Get network output
        y_val, _ = network(x_vec, Theta, model_type)
        y.append(y_val)
    
    # Plot network approximation with black line
    plt.plot(x, y, 'k-', linewidth=2, label=f"Network (N{model_type})")
    
    # Add grid, title, and legend
    plt.grid(True, alpha=0.3)
    plt.title(f"Model N{model_type}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show() 