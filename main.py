import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Data prep

# Dataset from 
# https://www.kaggle.com/code/nancyalaswad90/analysis-breast-cancer-prediction-dataset/input
df = pd.read_csv('data.csv')

print(df.describe())
print(df.columns)

df.drop(['id'], axis=1, inplace=True)

columns_list = list(df.columns)
columns_list.remove('diagnosis')

X = df.drop('diagnosis', axis=1).to_numpy()
y = LabelEncoder().fit_transform(df['diagnosis'])

# print(list(df.isnull().sum()))
# print(df.duplicated().sum())

train_x_temp, test1_x, train_y_temp, test1_y = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(train_x_temp, train_y_temp)

train_x, test2_x, train_y, test2_y = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10000, max_depth=10)
model.fit(train_x, train_y)

cv_scores = cross_val_score(model, X, y, cv=5)

predictions1 = model.predict(test1_x)
accuracy1 = np.mean(predictions1 == test1_y)

predictions2 = model.predict(test2_x)
accuracy2 = np.mean(predictions2 == test2_y)

print(f"Cross-Validation Accuracy: {np.mean(cv_scores)}")
print(f"Accuracy Test set 1: {accuracy1}")
print(f"Accuracy Test set 2: {accuracy2}")
print(f'Difference in accuracy test set 1: {accuracy1 - np.mean(cv_scores)}')
print(f'Difference in accuracy test set 2: {accuracy2 - np.mean(cv_scores)}')




