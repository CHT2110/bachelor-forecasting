import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv('forecasting.csv')

# List of categorical columns
cat_cols = ['ENTITY', 'PRODUCT_NAME']

# Convert categorical columns to category type
for col in cat_cols:
    df[col] = df[col].astype('category')

# Convert 'MONTH' to a datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df = df.set_index('MONTH')

# Extract relevant features from the datetime index
df['Year'] = df.index.year
df['Month'] = df.index.month

# Split the data into training and testing sets using data before October 1, 2023
train_data = df[df.index < '2023-04-01']
test_data = df[(df.index >= '2023-04-01')]

features = ['Year', 'Month', 'MEDIAN_PRICE', 'ENTITY', 'PRODUCT_NAME']
target = 'CHURNED_CUSTOMERS'

X_train = train_data[features]
Y_train = train_data[target]

X_test = test_data[features]
Y_test = test_data[target]

# Create an XGBoost regressor
model = xgb.XGBRegressor(enable_categorical = True)

# Fit the model on the training data
model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose=True, eval_metric = 'mae')


fi = pd.DataFrame(data=model.feature_importances_,
             index=model.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

test_data['prediction'] = model.predict(X_test)
df = df.merge(test_data[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['CHURNED_CUSTOMERS']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()
