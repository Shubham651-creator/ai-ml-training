import pandas as pd
from preprocessing import preprocess_data 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

X_train, X_test, y_train, y_test, feature_name = preprocess_data()

# Train model
# logistic_model = LogisticRegression(
#     class_weight="balanced",
#     max_iter=1000
# )

forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
forest_model.fit(X_train, y_train)

# Predict
y_pred = forest_model.predict(X_test)
print(y_pred)

# Evaluation Matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Evaluation
feature_importance = pd.DataFrame({
    "Feature": feature_name,
    "Importance": forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(15))