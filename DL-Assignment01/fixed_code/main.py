import numpy as np
import matplotlib.pyplot as plt
from network_functions import network, compute_gradient, gradient_descent, generate_datasets, plot

# Generate the datasets
X1, X2, X3 = generate_datasets()

# Define the number of neurons for each target function
K1 = 8   # For cos(3x), moderate complexity, oscillatory function
K2 = 6   # For e^(-x²), relatively simple bell-shaped function
K3 = 12  # For the 5th degree polynomial, higher complexity

# Initialize parameters for each network type and dataset
Theta_N1 = {
    "X1": (np.random.uniform(-1, 1, (2,)), None),
    "X2": (np.random.uniform(-1, 1, (2,)), None),
    "X3": (np.random.uniform(-1, 1, (2,)), None)
}

Theta_N2 = {
    "X1": (np.random.uniform(-1, 1, (2,)), None),
    "X2": (np.random.uniform(-1, 1, (2,)), None),
    "X3": (np.random.uniform(-1, 1, (2,)), None)
}

Theta_N3 = {
    "X1": (np.random.uniform(-1, 1, (K1, 2)), np.random.uniform(-1, 1, (K1+1,))),
    "X2": (np.random.uniform(-1, 1, (K2, 2)), np.random.uniform(-1, 1, (K2+1,))),
    "X3": (np.random.uniform(-1, 1, (K3, 2)), np.random.uniform(-1, 1, (K3+1,)))
}

# Learning rates and weight decay parameters
eta_values = {
    "N1": {"X1": 0.01, "X2": 0.01, "X3": 0.001},
    "N2": {"X1": 0.05, "X2": 0.05, "X3": 0.005},
    "N3": {"X1": 0.02, "X2": 0.02, "X3": 0.001}
}

# Weight decay parameter
lambda_value = 0.0001

# Store optimized parameters
Theta_N1_optimized = {}
Theta_N2_optimized = {}
Theta_N3_optimized = {}

# Run gradient descent for N1 (Linear Model)
print("Running gradient descent for N1...")
for dataset_name in ["X1", "X2", "X3"]:
    print(f"Processing {dataset_name}...")
    dataset = eval(dataset_name)  # Get the actual dataset
    initial_theta = Theta_N1[dataset_name]
    eta = eta_values["N1"][dataset_name]
    
    # Run gradient descent
    optimized_theta = gradient_descent(
        dataset, initial_theta, eta, model_type=1, lambda_=lambda_value
    )
    
    # Store optimized parameters
    Theta_N1_optimized[dataset_name] = optimized_theta
    
    # Print sample prediction after optimization
    x_sample = dataset[0][0]
    y_pred, _ = network(x_sample, optimized_theta, model_type=1)
    y_true = dataset[0][1]
    print(f"Sample prediction: {y_pred:.4f}, True value: {y_true:.4f}")

# Run gradient descent for N2 (One-layer with activation)
print("\nRunning gradient descent for N2...")
for dataset_name in ["X1", "X2", "X3"]:
    print(f"Processing {dataset_name}...")
    dataset = eval(dataset_name)  # Get the actual dataset
    initial_theta = Theta_N2[dataset_name]
    eta = eta_values["N2"][dataset_name]
    
    # Run gradient descent
    optimized_theta = gradient_descent(
        dataset, initial_theta, eta, model_type=2, lambda_=lambda_value
    )
    
    # Store optimized parameters
    Theta_N2_optimized[dataset_name] = optimized_theta
    
    # Print sample prediction after optimization
    x_sample = dataset[0][0]
    y_pred, _ = network(x_sample, optimized_theta, model_type=2)
    y_true = dataset[0][1]
    print(f"Sample prediction: {y_pred:.4f}, True value: {y_true:.4f}")

# Run gradient descent for N3 (Two-layer network)
print("\nRunning gradient descent for N3...")
for dataset_name in ["X1", "X2", "X3"]:
    print(f"Processing {dataset_name}...")
    dataset = eval(dataset_name)  # Get the actual dataset
    initial_theta = Theta_N3[dataset_name]
    eta = eta_values["N3"][dataset_name]
    
    # Run gradient descent
    optimized_theta = gradient_descent(
        dataset, initial_theta, eta, model_type=3, lambda_=lambda_value
    )
    
    # Store optimized parameters
    Theta_N3_optimized[dataset_name] = optimized_theta
    
    # Print sample prediction after optimization
    x_sample = dataset[0][0]
    y_pred, _ = network(x_sample, optimized_theta, model_type=3)
    y_true = dataset[0][1]
    print(f"Sample prediction: {y_pred:.4f}, True value: {y_true:.4f}")

print("\nOptimization complete for all models and datasets!")

# Plot results for each model and dataset
print("\nPlotting results...")

# Define plotting ranges
ranges = {
    "X1": [-3, 3],
    "X2": [-2, 2],
    "X3": [-5, 4]
}

# Define function names for titles
function_names = {
    "X1": "cos(3x)",
    "X2": "e^(-x²)",
    "X3": "x⁵ + 3x⁴ - 6x³ - 12x² + 5x + 129"
}

# Modified plotting function to display network approximation with black line
def custom_plot(X, Theta, model_type, R, dataset_name):
    """
    Plot the data samples and network approximation.
    
    Args:
        X: Dataset containing (x, t) pairs
        Theta: Network parameters (W1, w2)
        model_type: Type of network (1, 2, or 3)
        R: Range [min, max] for x-axis
        dataset_name: Name of the dataset for title
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
    plt.plot(x, y, 'k-', linewidth=2, label="Network (N{})".format(model_type))
    
    # Add true function for reference
    if dataset_name == "X1":
        true_y = [np.cos(3*x_val) for x_val in x]
        plt.plot(x, true_y, 'g--', alpha=0.7, label="True Function")
    elif dataset_name == "X2":
        true_y = [np.exp(-x_val**2) for x_val in x]
        plt.plot(x, true_y, 'g--', alpha=0.7, label="True Function")
    elif dataset_name == "X3":
        true_y = [x_val**5 + 3*x_val**4 - 6*x_val**3 - 12*x_val**2 + 5*x_val + 129 for x_val in x]
        plt.plot(x, true_y, 'g--', alpha=0.7, label="True Function")
    
    # Add grid, title, and legend
    plt.grid(True, alpha=0.3)
    plt.title(f"Model N{model_type} approximating {function_names[dataset_name]}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Plot N1 results in a single figure with subplots
print("\nPlotting N1 results...")
figure = plt.figure(figsize=(15, 5))

plt.subplot(131)
plot(X1, Theta_N1_optimized["X1"], 1, ranges["X1"])

plt.subplot(132)
plot(X2, Theta_N1_optimized["X2"], 1, ranges["X2"])

plt.subplot(133)
plot(X3, Theta_N1_optimized["X3"], 1, ranges["X3"])

plt.show()

# Plot N2 results in a single figure with subplots
print("\nPlotting N2 results...")
figure = plt.figure(figsize=(15, 5))

plt.subplot(131)
custom_plot(X1, Theta_N2_optimized["X1"], 2, ranges["X1"], "X1")

plt.subplot(132)
custom_plot(X2, Theta_N2_optimized["X2"], 2, ranges["X2"], "X2")

plt.subplot(133)
custom_plot(X3, Theta_N2_optimized["X3"], 2, ranges["X3"], "X3")

plt.show()

# Plot N3 results in a single figure with subplots
print("\nPlotting N3 results...")
figure = plt.figure(figsize=(15, 5))

plt.subplot(131)
plot(X1, Theta_N3_optimized["X1"], 3, ranges["X1"])

plt.subplot(132)
plot(X2, Theta_N3_optimized["X2"], 3, ranges["X2"])

plt.subplot(133)
plot(X3, Theta_N3_optimized["X3"], 3, ranges["X3"])

plt.show() 