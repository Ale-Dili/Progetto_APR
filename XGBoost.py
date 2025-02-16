from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# load data
X_test = np.load("data/processed_data_features/X_test.npy")
X_train = np.load("data/processed_data_features/X_train.npy")
X_val = np.load("data/processed_data_features/X_val.npy")
y_test = np.load("data/processed_data_features/y_test.npy")
y_train = np.load("data/processed_data_features/y_train.npy")
y_val = np.load("data/processed_data_features/y_val.npy")

# create model instance
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(eval_metric='mlogloss', random_state=44))
])


param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [ 0.1, 0.2],
    'clf__subsample': [0.7, 0.8, 1.0],
    'clf__colsample_bytree': [0.7, 0.8, 1.0],
    'clf__min_child_weight': [1, 3, 5]
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

# make predictions on test set
y_pred = model.predict(X_test)

# calculate metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
np.save('logs/XGBoost_no-crema/confusion_matrix.npy', cm)


print("Accuracy:", acc)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", cm)