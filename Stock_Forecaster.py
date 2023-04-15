
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create an input form to specify the ticker and period
st.sidebar.header('Input')
ticker = st.sidebar.text_input('Ticker', "AAPL")
period = st.sidebar.text_input('Period (e.g. 1y, 5d, 1mo)', "6mo")

# Load the stock data
ticker_data = yf.Ticker(ticker)
hist = ticker_data.history(period=period)
df = pd.DataFrame(hist)

# Split the data into features and target
X = df.drop(['Close'], axis=1)
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a function to train and evaluate a machine learning model
def train_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae, y_pred

# Create a sidebar to select the machine learning model
st.sidebar.header('Select Model')
model_name = st.sidebar.selectbox('', ['Linear Regression', 'Random Forest', 'Decision Tree', 'SVM', 'XGBoost'])

########################################
# Train and evaluate the selected models
########################################

# Train and evaluate the Linear Regression model
if model_name == 'Linear Regression':
    mae, y_pred = train_model(LinearRegression())
    st.write(f'Mean Absolute Error: {mae:.2f}')
    # Plot the predictions against the true values
    plt.scatter(y_test, y_pred)
    # Fit a line of best fit to the data
    coef = np.polyfit(y_test, y_pred, 1)
    poly_fit = np.poly1d(coef)
    plt.plot(y_test, poly_fit(y_test), 'r-')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.title('Linear Regression Predictions')
    # Display the plot in the main panel
    st.pyplot()

# Train and evaluate the Random Forest model
if model_name == 'Random Forest':
    mae, y_pred = train_model(RandomForestRegressor())
    st.write(f'Mean Absolute Error: {mae:.2f}')
    # Plot the predictions against the true values
    plt.scatter(y_test, y_pred)
    # Fit a line of best fit to the data
    coef = np.polyfit(y_test, y_pred, 1)
    poly_fit = np.poly1d(coef)
    plt.plot(y_test, poly_fit(y_test), 'r-')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.title('Random Forest Predictions')
    # Display the plot in the main panel
    st.pyplot()

# Train and evaluate the Decision Tree model
if model_name == 'Decision Tree':
    mae, y_pred = train_model(DecisionTreeRegressor())
    st.write(f'Mean Absolute Error: {mae:.2f}')
    # Plot the predictions against the true values
    plt.scatter(y_test, y_pred)
    # Fit a line of best fit to the data
    coef = np.polyfit(y_test, y_pred, 1)
    poly_fit = np.poly1d(coef)
    plt.plot(y_test, poly_fit(y_test), 'r-')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.title('Decision Tree Predictions')
    # Display the plot in the main panel
    st.pyplot()

# Train and evaluate the SVM model
if model_name == 'SVM':
    mae, y_pred = train_model(SVR())
    st.write(f'Mean Absolute Error: {mae:.2f}')
    # Plot the predictions against the true values
    plt.scatter(y_test, y_pred)
    # Fit a line of best fit to the data
    coef = np.polyfit(y_test, y_pred, 1)
    poly_fit = np.poly1d(coef)
    plt.plot(y_test, poly_fit(y_test), 'r-')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.title('SVM Predictions')
    # Display the plot in the main panel
    st.pyplot()
    
# Train and evaluate the XGBoost model
if model_name == 'XGBoost':
    mae, y_pred = train_model(XGBRegressor())
    st.write(f'Mean Absolute Error: {mae:.2f}')
    # Plot the predictions against the true values
    plt.scatter(y_test, y_pred)
    # Fit a line of best fit to the data
    coef = np.polyfit(y_test, y_pred, 1)
    poly_fit = np.poly1d(coef)
    plt.plot(y_test, poly_fit(y_test), 'r-')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.title('XGBoost Predictions')
    # Display the plot in the main panel
    st.pyplot()    
    
# Additional Visualization

# Plot the stock data as a line chart
plt.plot(df.index, df['Close'])
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Stock Data')
st.pyplot()

# Show the distribution of the true values and predictions
sns.distplot(y_test, label='True values')
sns.distplot(y_pred, label='Predictions')
plt.xlabel('Close')
plt.ylabel('Density')
plt.title('Distribution of True Values and Predictions')
plt.legend()
st.pyplot()

# Add a vertical line at the mean of the true values and predictions
plt.axvline(y_test.mean(), color='b', linestyle='dashed', linewidth=2)
plt.axvline(y_pred.mean(), color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Close')
plt.ylabel('Density')
plt.title('Mean of True Values and Predictions')
st.pyplot()

# Add a horizontal line at the mean of the true values
plt.axhline(y_test.mean(), color='b', linestyle='dashed', linewidth=2)
plt.xlabel('Predictions')
plt.ylabel('True values')
plt.title('Mean of True Values')
st.pyplot()

# Use a scatterplot with a regression line to show the relationship between the true values and predictions
sns.scatterplot(y_test, y_pred)
sns.regplot(y_test, y_pred)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
st.pyplot()
