import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load data
X_test = np.load("data/processed_data_features/X_test.npy")
X_train = np.load("data/processed_data_features/X_train.npy")
X_val = np.load("data/processed_data_features/X_val.npy")
y_test = np.load("data/processed_data_features/y_test.npy")
y_train = np.load("data/processed_data_features/y_train.npy")
y_val = np.load("data/processed_data_features/y_val.npy")

# define parameter grid for the RandomForestClassifier
param_grid = {
    'n_estimators': [600, 800, 1200, 1400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [1, 2, 5],
    'min_samples_leaf': [None, 1, 2]
}



#print train shape
print(X_train.shape)
print(y_train.shape)



# create a RandomForestClassifier with a fixed random_state
rf = RandomForestClassifier(random_state=44)

# initialize GridSearchCV with 5-fold cross validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# use the best estimator from grid search for further evaluation
best_rf = grid_search.best_estimator_

# evaluate on the validation set
y_val_pred = best_rf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

# evaluate on the test set
y_test_pred = best_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))



cm = confusion_matrix(y_test, y_test_pred)

np.save('logs/decision_tree_crema/confusion_matrix.npy', cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()