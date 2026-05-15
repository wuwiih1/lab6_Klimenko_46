import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('insurance.csv')

# 1)Виведіть загальну інформацію про зазначену базу даних.
print(df.info())
# 2) Адаптуйте дані до використання лінійної регресії.
df = pd.get_dummies(df, columns=['region'], drop_first=True)
# 3) Розділіть дані на тестову та навчальну вибірки.
region_cols = [col for col in df.columns if 'region_' in col]
feature_cols = ['age', 'bmi'] + region_cols

X = df[feature_cols]
y = df['expenses']  # Виправлено з 'charges' на 'expenses'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Побудуйте модель машинного навчання, яка дозволяє
# прогнозувати вартість страхування в залежності від віку, індексу маси
# тіла та регіону, де людина проживає.
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
# 5) Порівняйте спрогнозовані результати із реальними даними.
y_pred = model.predict(X_test)
# Знайдіть різницю між реальними даними тестової вибірки та
# прогнозованими, зокрема обчисліть, який відсоток становить абсолютне
# значення похибки до реальних даних.
error_percentage = np.abs(y_test - y_pred) / y_test * 100
# 6) Побудуйте гістограму відсоткових значень похибки для
# кожного прогнозованого результату.
plt.hist(error_percentage, bins=20, edgecolor='black')
plt.xlabel('Percentage Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()

exit()

