import pandas as pd

from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("dataset/gestures.csv")


# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Display results
print("Total samples:", len(X))

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())