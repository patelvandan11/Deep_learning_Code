import numpy as np
import pandas as pd

# Parameters for data generation
num_samples = 1000  # Number of data points
num_features = 10   # Number of features (unshaped data)

# Generate random data (unshaped)
data = np.random.rand(num_samples, num_features)

# Generate corresponding labels (binary classification example)
labels = np.random.randint(0, 2, size=(num_samples, 1))

# Combine features and labels into one array
data_with_labels = np.hstack((data, labels))

# Create a DataFrame for easier handling
columns = [f"Feature_{i+1}" for i in range(num_features)] + ["Label"]
df = pd.DataFrame(data_with_labels, columns=columns)

# Save the data to a CSV file
output_file = "unshaped_data.csv"
df.to_csv(output_file, index=False)

print(f"Data successfully generated and saved to {output_file}")