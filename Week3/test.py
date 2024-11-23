import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generating random time series data
n_points = 100
n_features = 10
X = np.random.randn(n_points, n_features)
y = np.random.randint(0, 2, size=n_points)  # Binary target variable

# Set the number of splits
n_splits = 5
k_fold_size = n_points // n_splits

best_score = float('-inf')
best_params = None

# Hyperparameter grid for tuning
n_estimators_values = [10, 50]
max_depth_values = [None, 5]

# Perform blocked cross-validation
for i in range(n_splits):
    start_train = i * k_fold_size
    end_train = start_train + k_fold_size
    
    # Ensure we don't go out of bounds for the last split
    if i == n_splits - 1:
        end_train = n_points
        
    train_indices = np.arange(start_train, end_train)
    val_indices = np.arange(end_train, min(end_train + k_fold_size, n_points))
    
    if len(val_indices) > 0:
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Grid search over parameters
        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)

                # Update best parameters based on score
                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}

print("Best parameters found: ", best_params)
