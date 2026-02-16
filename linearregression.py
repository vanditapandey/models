#Simple Linear Regression (House Prices)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Small house price dataset
data = {
    'area':  [650, 800, 900, 1200, 1500, 1700, 1850, 2000, 2200, 2500],
    'rooms': [2,   2,   3,   3,    3,    4,    4,    4,    5,    5],
    'price': [35, 42, 48, 60, 72, 80, 88, 92, 105, 120]  # in lakhs
}
df = pd.DataFrame(data)
print(df.head())

# 2. Use area as single feature
X = df[['area']]
y = df['price']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 3. Evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 4. Scatter plot + regression line
plt.scatter(X, y, color='blue', label='Actual data')
x_line = pd.DataFrame({'area': sorted(df['area'])})
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (lakhs)')
plt.title('Area vs House Price (Simple Linear Regression)')
plt.legend()
plt.show()
