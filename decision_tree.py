import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("../data/FD001_feature_engineered.csv")

X = df.drop(columns=["RUL", "unit", "max_cycle"])
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Decision Tree model trained.")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Decision Tree Results")
print("--------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)