import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv('forecasting.csv')

# List of categorical columns
cat_cols = ['ENTITY', 'PRODUCT_NAME']

# Convert 'MONTH' to a datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df = df.set_index('MONTH')

# Convert categorical columns to category type after handling missing values
for col in cat_cols:
    df[col] = df[col].astype('category')

# Extract relevant features from the datetime index
df['Year'] = df.index.year
df['Month'] = df.index.month

# Split the data into training and testing sets using data before October 1, 2023
train_data = df[df.index < '2023-04-01']
test_data = df[(df.index >= '2023-04-01')]

# Define features and target variable
features = ['Year', 'Month', 'MEDIAN_PRICE', 'ENTITY', 'PRODUCT_NAME']
target = 'CHURNED_CUSTOMERS'

# Separate training and testing data
X_train = train_data[features]
Y_train = train_data[target]

X_test = test_data[features]
Y_test = test_data[target]

# Convert categorical columns to category type in training and testing data
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Create an XGBoost regressor with enable_categorical set to True
model = xgb.XGBRegressor(enable_categorical=True)

# Fit the model on the training data
model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose=True, eval_metric='mae')

# Dictionary with individual median prices for each combination (increased by 8%)
median_prices_to_change = {
    ('advisor', 'advisor'): 216,
    ('billy', 'pro'): 27,
    ('billy', 'premium'): 37,
    ('billy', 'subscription'): 33,
    ('billy', 'annual report'): 20,
    ('billy', 'accounting package'): 536,
    ('kontist', 'duo'): 12,
    ('kontist', 'kontax'): 179,
    ('kontist', 'accounting'): 29,
    ('kontist', 'premium'): 9,
    ('kontist', 'annual report'): 25,
    ('lending', 'loan'): 115,
    ('salary', 'basic'): 0,
    ('salary', 'premium'): 10,
    ('tellow', 'plus'): 26,
    ('tellow', 'compleet'): 70,
    ('tellow', 'banking'): 1,
    ('tellow', 'basis'): 15,
    ('zervant', 'custom'): 155,
    ('zervant', 'pro'): 17,
    ('zervant', 'growth'): 40,
    ('zervant', 'starter'): 9,
    ('zervant', 'merchant'): 82
}

# Increase all prices by 8%
median_prices_to_change_8percent = {key: value * 1.08 for key, value in median_prices_to_change.items()}

# Function to predict churned customers for a given median price scenario
def predict_churned_customers(model, entity, product_name, median_price):
    # Create a DataFrame for prediction input
    input_data = pd.DataFrame({
        'Year': [2023],  # Adjust the year as needed
        'Month': [4],    # Adjust the starting month as needed
        'MEDIAN_PRICE': [median_price],
        'ENTITY': [entity],
        'PRODUCT_NAME': [product_name]
    })

    # Convert categorical columns to category type in the prediction data
    for col in cat_cols:
        input_data[col] = input_data[col].astype('category')

    # Predict churned customers for the given input
    prediction = model.predict(input_data)

    return prediction[0]

# Initialize a subplot for each combination
fig, axs = plt.subplots(len(median_prices_to_change_8percent), 1, figsize=(10, 6 * len(median_prices_to_change_8percent)))

# Iterate through the dictionary and perform predictions
for i, ((entity_to_change, product_name_to_change), new_median_price) in enumerate(median_prices_to_change_8percent.items()):
    current_median_price = df[(df['ENTITY'] == entity_to_change) & (df['PRODUCT_NAME'] == product_name_to_change)]['MEDIAN_PRICE'].values[0]

    # Predict churned customers for the current median price
    current_prediction = predict_churned_customers(model, entity_to_change, product_name_to_change, current_median_price)

    # Predict churned customers for the new median price
    new_prediction = predict_churned_customers(model, entity_to_change, product_name_to_change, new_median_price)

    # Add 'prediction' column to test_data for the current combination
    test_data = test_data.copy()
    test_data.loc[(test_data['ENTITY'] == entity_to_change) & (test_data['PRODUCT_NAME'] == product_name_to_change), 'prediction'] = current_prediction


    # Plot the results
    axs[i].plot(test_data[test_data['ENTITY'] == entity_to_change][test_data['PRODUCT_NAME'] == product_name_to_change].index, test_data[test_data['ENTITY'] == entity_to_change][test_data['PRODUCT_NAME'] == product_name_to_change]['CHURNED_CUSTOMERS'], label='Truth Data', marker='o')
    axs[i].plot(test_data[test_data['ENTITY'] == entity_to_change][test_data['PRODUCT_NAME'] == product_name_to_change].index, test_data[test_data['ENTITY'] == entity_to_change][test_data['PRODUCT_NAME'] == product_name_to_change]['prediction'], label='Predictions (Current Median Price)', linestyle='--', marker='.')
    axs[i].plot(test_data[test_data['ENTITY'] == entity_to_change][test_data['PRODUCT_NAME'] == product_name_to_change].index, new_prediction, label=f'Predictions (New Median Price: {new_median_price})', linestyle='--', marker='.')


    axs[i].set_title(f"Entity: {entity_to_change}, Product: {product_name_to_change}")
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('Churned Customers')
    axs[i].legend()

plt.tight_layout()
plt.show()

