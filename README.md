# Stock Price Forecasting Tool


---


---

Create a tool that allows users to select and train a machine learning model to predict stock closing prices. The tool allows users to input a ticker symbol and a time period, and it retrieves stock data for the specified ticker and period from Yahoo Finance. The tool then trains and evaluates a machine learning model on the stock data, and it displays a plot of the model's predictions against the true values.

The available machine learning models are:

Linear Regression
Random Forest
Decision Tree
Support Vector Machine (SVM)
XGBoost

The app calculates the mean absolute error (MAE) of the model's predictions and displays it to the user.

Import libraries:

streamlit

yfinance

pandas

numpy

seaborn

matplotlib

scikit-learn (sklearn)

xgboost 

---

To run the app, open a terminal and navigate to the directory where the code is saved. Then, enter the command streamlit run StockPricePredictor.py. This will launch the app in your default web browser.

---

### Usage

To use the tool:
Enter a ticker symbol and a period of time in the input form on the sidebar. 
Select a machine learning model from the dropdown menu. 
The tool will display the mean absolute error and a scatter plot of the predictions against the true values.
