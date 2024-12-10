import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Generate feature data for two classes
num_samples = 100
class_0_x = np.random.randn(num_samples // 2) - 1  # Class 0, shifted
class_0_y = np.random.randn(num_samples // 2) - 0.5

class_1_x = np.random.randn(num_samples // 2) + 1  # Class 1, shifted
class_1_y = np.random.randn(num_samples // 2) + 0.5

# Combine the data
X = np.concatenate([np.column_stack((class_0_x, class_0_y)),
                    np.column_stack((class_1_x, class_1_y))], axis=0)

# Labels: 0 for class 0, 1 for class 1
y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

# Create a DataFrame
data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
data['Label'] = y

# Save the DataFrame to a CSV file
data.to_csv('classification_data.csv', index=False)

# Plotting the generated data
plt.figure(figsize=(8, 6))
plt.scatter(data[data['Label'] == 0]['Feature1'], data[data['Label'] == 0]['Feature2'], color='yellow', label='Class 0')
plt.scatter(data[data['Label'] == 1]['Feature1'], data[data['Label'] == 1]['Feature2'], color='purple', label='Class 1')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.title('Generated Classification Data')
plt.show()

print("Data has been saved to 'classification_data.csv'.")
