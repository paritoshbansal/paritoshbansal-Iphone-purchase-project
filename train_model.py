
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
path = "iphone_purchase_records.csv"
target_column = "Purchase Iphone"
data = pd.read_csv(path)
df = data.copy()

# Encode categorical variable
lb = LabelEncoder()
df["Gender"] = lb.fit_transform(df["Gender"])

# Features and Target
x = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# KNN Error Tracking
error = []
k_values = []
best_k = 1
min_error = float('inf')

for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    model = knn.fit(x_train, y_train)
    pred_i = model.predict(x_test)
    err = 1 - accuracy_score(y_test, pred_i)
    error.append(err)
    k_values.append(i)
    
    if err < min_error:
        min_error = err
        best_k = i

# Train final model with best k
knn_final = KNeighborsClassifier(n_neighbors=best_k)
model_final = knn_final.fit(x_train, y_train)

# Save model and label encoder
joblib.dump(model_final, "model.pkl")
joblib.dump(lb, "label_encoder.pkl")

print(f"Best k: {best_k}")
