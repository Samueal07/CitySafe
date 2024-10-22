# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
df = pd.read_csv('new_dataset.csv')

# Preprocess the data
# Map the cities and crime types to numeric values
df['City'] = df['City'].astype('category').cat.codes
df['Type'] = df['Type'].astype('category').cat.codes

# Select the feature columns and target column
X = df[['Year', 'City', 'Population', 'Type']]  # Features
y = df['Crime Rate']  # Target variable

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model to a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'model.pkl'")
