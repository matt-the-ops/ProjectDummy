import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load captured dataset
df = pd.read_csv('fsl_dataset.csv')
X = df.drop('label', axis=1)
y = df['label']

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluate accuracy
y_pred = model.predict(X_test)
print(f"Training Complete! Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 5. Save model weights to file
with open('fsl_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved model file as 'fsl_model.pkl'.")