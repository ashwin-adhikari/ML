from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

# Load a sample dataset
data = load_iris()
X, y = data.data, data.target

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=40, shuffle=True, random_state=1)

# Initialize a list to store results
accuracies = []

# Perform Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X, y):
    # Split data into train and test for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train the model
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Calculate average accuracy
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {average_accuracy:.3f}')
