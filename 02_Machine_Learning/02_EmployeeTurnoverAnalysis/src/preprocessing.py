import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.DataFrame(data)

# Remove useless column based on EDA.py
df = df.drop(
    columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "StandardHours"
    ]
)

# Encode target
print(df["Attrition"].unique())
df["Attrition"] = (
    df["Attrition"]
      .map({"Yes":1, "No":0})
)
print(df["Attrition"].unique()) 

# One-Hot encoding
print(
    df.select_dtypes(
        include="str"
    ).columns
)

df = pd.get_dummies(
    df,
    drop_first=True
)
print(df)

# X & y
y = df["Attrition"]
X = df.drop(
    columns=["Attrition"]
)

print(X.shape)
print(y.shape)

stratify=y
# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)   
X_test = scaler.transform(
    X_test
)

def preprocess_data():
    return X_train, X_test, y_train, y_test, X.columns