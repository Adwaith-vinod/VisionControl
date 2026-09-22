import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load dataset
df = pd.read_csv("dataset/gestures.csv")


# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
print("Training model...")

model.fit(X_train, y_train)

print("Training complete!")


# Save trained model
model_path = "models/gesture_model.pkl"

joblib.dump(model, model_path)

print("Model saved to:", model_path)


# Predict gestures for test data
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print()
print("Accuracy:", accuracy)


# Classification report
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))