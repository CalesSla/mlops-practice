import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

with open("config.json") as f:
    config = json.load(f)

df = pd.read_csv(config["data"]["raw"])
df.dropna(inplace=True)
df["wind_humidity_ratio"] = df["wind_speed"] / df["humidity"].replace(0, 0.1)

X = df[config["features"]["selected"]]
y = df[config["target"]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["model"]["test_size"],
    random_state=config["model"]["random_state"],
)

model = LinearRegression()
model.fit(X_train, y_train)

model_path = Path(config["model"]["output_path"])
model_path.parent.mkdir(parents=True, exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(model, f)

score = model.score(X_test, y_test)
print(f"Model R^2 score: {score}")
