# Step 1: Data Collection
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Selection
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Step 6: Prediction Function
def predict_cancer(input_data):
    """
    Predicts if a person has breast cancer based on input data.

    Parameters:
    input_data (list or array-like): Input features for prediction.

    Returns:
    str: Prediction result indicating whether the person has cancer or not.
    """
    # Scale the input data
    input_data = scaler.transform([input_data])

    # Predict the class
    prediction = model.predict(input_data)

    # Interpret the result
    if prediction[0] == 1:
        return "The person has breast cancer."
    else:
        return "The person does not have breast cancer."


# Example Usage
# Replace example_input with the actual feature values for a new sample
example_input = X_test[0]  # Using the first sample from the test set for demonstration
print(predict_cancer(example_input))