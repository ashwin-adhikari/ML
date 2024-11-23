
## Assignment Notes

### Correlation matrix
- Positive correlation indicates that as one variable increases, the other tends to increase. A negative correlation indicates that as one variable increases, the other tends to decrease.
- The values range from -1 to 1, where 1 is a perfect positive correlation, -1 is a perfect negative correlation, and 0 indicates no correlation.

These correlations provide valuable insights for predictive modeling and understanding the factors that influence wine quality. It is essential to consider these relationships in further analyses and model building.


### Confusion matrix
```python
Confusion Matrix:
 [[ 3  3  0  0  0]
 [ 0 96  0  0  0]
 [ 0  0 99  0  0]
 [ 0  0  0 26  0]
 [ 0  0  0  2  0]]
 ```
 
#### Interpretation
- Diagonal (True Positives and True Negatives):

    - Values along the diagonal (3, 96, 99, 26, 0) represent correct predictions.

    - Class 1: 3 instances correctly predicted.

    - Class 2: 96 instances correctly predicted.

    - Class 3: 99 instances correctly predicted.

    - Class 4: 26 instances correctly predicted.

    - Class 5: No instances correctly predicted (0).

- Off-Diagonal (False Positives and False Negatives):

    - Non-diagonal elements show misclassifications.

    - Class 1: 3 instances misclassified as class 2.

    - Class 5: 2 instances misclassified as class 4.

#### Statistical Analysis

- Class 1: 50% correctly identified (3 out of 6).

- Class 2: 100% correctly identified (96 out of 96).

- Class 3: 100% correctly identified (99 out of 99).

- Class 4: 100% correctly identified (26 out of 26).

- Class 5: 0% correctly identified (0 out of 2).


### Classification Report
```
Classification Report:
               precision    recall  f1-score   support

           4       1.00      0.50      0.67         6
           5       0.97      1.00      0.98        96
           6       1.00      1.00      1.00        99
           7       0.93      1.00      0.96        26
           8       0.00      0.00      0.00         2

    accuracy                           0.98       229
   macro avg       0.78      0.70      0.72       229
weighted avg       0.97      0.98      0.97       229
```
#### Classification Report Analysis
The classification report provides detailed metrics for each class, along with overall model performance metrics. Here’s a statistical breakdown:

##### Class-wise Performance
- Class 4

    - Precision: 1.00 (Model correctly identified all its predictions as class 4)

    - Recall: 0.50 (Model correctly identified 50% of actual class 4 instances)

    - F1-Score: 0.67 (Moderate performance)

    - Support: 6 (Actual occurrences in the dataset)

- Class 5

    - Precision: 0.97 (High precision, few false positives)

    - Recall: 1.00 (Perfect recall)

    - F1-Score: 0.98 (Excellent performance)

    - Support: 96

- Class 6

    - Precision: 1.00 (Perfect precision)

    - Recall: 1.00 (Perfect recall)

    - F1-Score: 1.00 (Excellent performance)

    - Support: 99

- Class 7

    - Precision: 0.93 (High precision)

    - Recall: 1.00 (Perfect recall)

    - F1-Score: 0.96 (Excellent performance)

    - Support: 26

- Class 8

    - Precision: 0.00 (Low precision, high false positives or no correct predictions)

    - Recall: 0.00 (No correct identifications)

    - F1-Score: 0.00 (Poor performance)

    - Support: 2

- Overall Performance

    - Accuracy: 0.98 (98% of predictions were correct)

    1. Macro Average

        - Precision: 0.78

        - Recall: 0.70

        - F1-Score: 0.72

    2. Weighted Average

        - Precision: 0.97

        - Recall: 0.98

        - F1-Score: 0.97

### Cross Validation Score
```python
Cross-Validation Scores: [0.9782, 0.9825, 0.9782, 0.9825, 0.9737]
```
- The scores are relatively consistent, with a slight variation. The lowest score is 0.9345, and the highest is 0.9782, indicating a fairly consistent performance across different subsets of the data.
- All scores are high (above 0.93), suggesting that the model is performing well across all folds. 
- The small range of scores (from 0.9345 to 0.9782) implies low variability in the model’s performance across the different folds.