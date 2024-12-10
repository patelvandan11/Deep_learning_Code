import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Generate sample data
num_samples = 1000
feature1 = np.random.randn(num_samples)  # Random data for feature 1
feature2 = np.random.randn(num_samples)  # Random data for feature 2

# Generate labels (binary classification: 0 or 1)
labels = np.random.choice([0, 1], size=num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Feature1': feature1,
    'Feature2': feature2,
    'Label': labels
})

# Save the DataFrame to a CSV file
data.to_csv('classi_multiple.csv', index=False)

print("Data has been saved to 'classification_data.csv'.")
