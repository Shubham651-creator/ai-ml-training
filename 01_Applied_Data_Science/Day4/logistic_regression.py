import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = {
    "Hours_studied" : [1,2,3,4,5,6,7,8],
    "Pass": [0,0,0,1,1,1,1,1]
}
dFrame = pd.DataFrame(data)
print(dFrame)

# Predication = x.w + b
x = dFrame[["Hours_studied"]]
y = dFrame["Pass"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

# Train model
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)

# Predict
y_pred = logistic_model.predict(x_test)
print(y_pred)

# Evulation Matrix
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))