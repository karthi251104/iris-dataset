import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = joblib.load("model/model.joblib")
predictions = model.predict(X)
print("Accuracy:", accuracy_score(y, predictions))
