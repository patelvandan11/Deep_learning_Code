import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate U-shaped data
def generate_ushape_data(n_points=500, noise=0.1):
    """
    Generate U-shaped data for neural network training.
    :param n_points: Number of data points to generate.
    :param noise: Standard deviation of Gaussian noise.
    :return: Tuple of arrays (X, y).
    """
    # Generate x-coordinates
    x1 = np.linspace(-2, 2, n_points // 2)
    x2 = np.linspace(2, -2, n_points // 2)
    
    # Generate y-coordinates for U-shape
    y1 = x1**2
    y2 = -x2**2 + 4

    # Combine x and y for both parts of the U-shape
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    # Add Gaussian noise
    x += np.random.normal(0, noise, x.shape)
    y += np.random.normal(0, noise, y.shape)

    return x, y

# Generate data
n_points = 500
noise_level = 0.05
x, y = generate_ushape_data(n_points=n_points, noise=noise_level)

# Create a DataFrame
data = pd.DataFrame({'X': x, 'Y': y})

# Save to CSV file
csv_filename = 'ushape_data.csv'
data.to_csv(csv_filename, index=False)

print(f"U-shaped data saved to {csv_filename}")

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, edgecolor='k')
plt.title("U-shaped Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()