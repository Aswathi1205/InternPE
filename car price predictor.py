import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Generate synthetic data with more features
np.random.seed(42)
num_samples = 100
years = np.random.randint(2000, 2022, num_samples)
mileages = np.random.randint(5000, 150000, num_samples)
brands = np.random.choice(['Toyota', 'Ford', 'BMW', 'Audi'], num_samples)
engine_sizes = np.random.choice([1.2, 1.6, 2.0, 2.5], num_samples)

# Create a realistic price
prices = 20000 - (2022 - years) * 1000 - mileages * 0.05 + np.random.normal(0, 2000, num_samples)

data = pd.DataFrame({
    'Year': years,
    'Mileage': mileages,
    'Brand': brands,
    'Engine Size': engine_sizes,
    'Price': prices
})

# Feature Engineering
data['Car Age'] = 2022 - data['Year']

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Brand'])

# Select features and target
X = data.drop(columns=['Price', 'Year'])  # Drop 'Year' as we use 'Car Age' instead
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Ridge regression model
model = Ridge()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example of predicting a new car's price
new_car = pd.DataFrame({
    'Mileage': [15000],
    'Engine Size': [2.0],
    'Car Age': [2],
    'Brand_Audi': [0],
    'Brand_BMW': [0],
    'Brand_Ford': [0],
    'Brand_Toyota': [1]
})
predicted_price = model.predict(new_car)
print(f'Predicted Price: {predicted_price[0]}')