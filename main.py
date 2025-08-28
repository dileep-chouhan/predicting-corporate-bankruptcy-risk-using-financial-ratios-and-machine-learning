import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic financial data
num_companies = 1000
data = {
    'CurrentRatio': np.random.uniform(0.5, 3, num_companies),
    'DebtEquityRatio': np.random.uniform(0, 2, num_companies),
    'ROA': np.random.uniform(-0.1, 0.2, num_companies),
    'Bankrupt': np.random.choice([0, 1], num_companies, p=[0.9, 0.1]) # 10% bankruptcy rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data.  In real-world scenarios, this step would be crucial.
# --- 3. Data Splitting ---
X = df[['CurrentRatio', 'DebtEquityRatio', 'ROA']]
y = df['Bankrupt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 4. Model Training ---
model = LogisticRegression()
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# --- 6. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Bankrupt', 'Bankrupt'], 
            yticklabels=['Not Bankrupt', 'Bankrupt'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
# Save the plot to a file
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
plt.scatter(df['DebtEquityRatio'], df['ROA'], c=df['Bankrupt'], cmap='viridis')
plt.xlabel('Debt to Equity Ratio')
plt.ylabel('Return on Assets (ROA)')
plt.title('Bankruptcy Prediction')
plt.colorbar(label='Bankruptcy (0: No, 1: Yes)')
output_filename2 = 'scatter_plot.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")