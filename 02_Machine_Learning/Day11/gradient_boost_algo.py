import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8],
    "Pass": [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)
print(df)

X = df[["Hours_Studied"]]
y = df["Pass"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train
model = GradientBoostingClassifier(
    n_estimators = 10,
    random_state=42
)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
# y_pred = model.predict_proba(X_test)

print("Predictions:", y_pred)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:",
      precision_score(y_test, y_pred))

print("Recall:",
      recall_score(y_test, y_pred))

print("F1:",
      f1_score(y_test, y_pred))