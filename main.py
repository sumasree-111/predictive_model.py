from sklearn.linear_model import LinearRegression
import pandas as pd

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [20, 35, 45, 55, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[9]])

print("Predicted Marks:", prediction[0])
