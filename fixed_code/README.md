# Universal Function Approximator

This code implements three different neural network architectures to approximate various functions:

1. **N1**: One-layer network (linear transformation only)
2. **N2**: One-layer network with non-linear activation function (tanh)
3. **N3**: Two-layer network (hidden layer with non-linear activation function)

## Functions to Approximate

The networks are trained to approximate three different functions:

1. **X1**: t = cos(3x) for x ∈ [-2,2]
2. **X2**: t = e^(-x²) for x ∈ [-1,1]
3. **X3**: t = x^5 + 3x^4 - 6x^3 - 12x^2 + 5x + 129 for x ∈ [-4,2.5]

## Code Structure

- **network_functions.py**: Contains all the core functions:
  - `network()`: Implements the three network architectures
  - `compute_gradient()`: Calculates gradients for backpropagation
  - `gradient_descent()`: Performs the optimization
  - `generate_datasets()`: Creates the three datasets
  - `plot()`: Visualizes the results

- **main.py**: Main script that runs the training and visualization

## How to Run

1. Make sure you have the required dependencies installed:
   ```
   pip install numpy matplotlib
   ```

2. Run the main script:
   ```
   python main.py
   ```

3. The script will:
   - Generate the datasets
   - Train all three network types on all three functions
   - Display plots showing the approximation results

## Implementation Details

- **Gradient Descent**: Uses 10,000 epochs with different learning rates for each model and dataset
- **Regularization**: L2 regularization (weight decay) is applied with λ = 0.0001
- **Hidden Neurons**: 
  - 8 neurons for X1 (cos function)
  - 6 neurons for X2 (exponential function)
  - 12 neurons for X3 (polynomial function)

## Expected Results

- **N1** (Linear): Can only approximate linear relationships, so it will perform poorly on all three functions
- **N2** (One-layer with activation): Can approximate simple non-linear functions (X1, X2) but struggles with complex ones (X3)
- **N3** (Two-layer): Can approximate all three functions well, demonstrating the Universal Approximation Theorem 