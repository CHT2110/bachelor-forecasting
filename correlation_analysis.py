import pandas as pd
from sklearn.linear_model import LinearRegression

# Read data from CSV file
df = pd.read_csv('customer_elasticity.csv')  # Replace 'your_data.csv' with your actual CSV file

# Perform linear regression and store results in a DataFrame
results_regression = pd.DataFrame(columns=['entity', 'product_name', 'slope', 'intercept'])

# Perform correlation analysis and store results in a separate DataFrame
results_correlation = pd.DataFrame(columns=['entity', 'product_name', 'correlation'])

# Group data by entity and product_name
grouped_data = df.groupby(['ENTITY', 'PRODUCT_NAME'])

# Perform linear regression and correlation analysis for each group
for group, data in grouped_data:
    X = data[['MEDIAN_PRICE']]
    y = data['CHURN_RATE']
    
    # Check if there are enough data points for regression
    if len(X) > 1:
        # Perform linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Get regression coefficients
        slope = model.coef_[0]
        intercept = model.intercept_

        # Append regression results to the DataFrame
        results_regression = results_regression._append({
            'entity': group[0],
            'product_name': group[1],
            'slope': slope,
            'intercept': intercept
        }, ignore_index=True)

    # Perform correlation analysis
    correlation = data[['CHURN_RATE', 'MEDIAN_PRICE']].corr().iloc[0, 1]

    # Append correlation results to the DataFrame
    results_correlation = results_correlation._append({
        'entity': group[0],
        'product_name': group[1],
        'correlation': correlation
    }, ignore_index=True)

# Print the regression results
print("Regression Results:")
print(results_regression)

# Print the correlation results
print("\nCorrelation Results:")
print(results_correlation)