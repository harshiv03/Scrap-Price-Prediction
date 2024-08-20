import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import SGDRegressor  
from sklearn.metrics import mean_squared_error, r2_score  
import csv  
 
# Define the required features and target column  
X_features = ['Price with Overhead Cost Per MT', 'Grade', 'Alang Scrap Price', 'Fresh Procurement Price']  
target_column = 'Last Selling Price'  
 
# Load the CSV file using pandas  
df = pd.read_csv('/content/HeavyMS.csv')  
 
# Ensure that the required columns are present in the dataframe  
required_columns = X_features + [target_column]  
df = df[required_columns]  
 
# Separate features and target  
x = df[X_features].values  
y = df[target_column].values  
 
# Split the data into training and testing sets  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)  
 
# Remove rows with NaN values  
x_train = x_train[~np.isnan(x_train).any(axis=1)]  
y_train = y_train[~np.isnan(y_train)]  
x_test = x_test[~np.isnan(x_test).any(axis=1)]  
y_test = y_test[~np.isnan(y_test)]  
 
# Print data shapes and types  
print(f"X Shape: {x_train.shape}, X Type: {type(x_train)}")  
print(x_train)  
print(f"y Shape: {y_train.shape}, y Type: {type(y_train)}")  
print(y_train)  
 
# Normalize the data  
scaler = StandardScaler()  
X_norm = scaler.fit_transform(x_train)  
X_norm_test = scaler.transform(x_test)  
 
print(f"Peak to Peak range by column in Raw X: {np.ptp(x_train, axis=0)}")  
print(f"Peak to Peak range by column in Normalized X: {np.ptp(X_norm, axis=0)}")  
 
# Train the SGDRegressor  
sgdr = SGDRegressor(max_iter=1000)  
sgdr.fit(X_norm, y_train)  
print(sgdr)  
print(f"Number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")  
 
# Model parameters  
b_norm = sgdr.intercept_  
w_norm = sgdr.coef_  
print(f"Model parameters: w: {w_norm}, b: {b_norm}")  
 
# Make predictions  
y_pred_sgd = sgdr.predict(X_norm)  
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"Prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")  
print(f"Prediction on training set:\n{y_pred[:4]}")  
print(f"Target values \n{y_train[:4]}")  
 
# Print the accuracy metrics  
mse = mean_squared_error(y_train, y_pred_sgd)  
r2 = r2_score(y_train, y_pred_sgd)  
print(f"Mean Squared Error (MSE): {mse:.2f}")  
print(f"R-squared (R2): {r2:.2f}")  
 
# Print the correlation matrix  
corr_matrix = np.corrcoef(X_norm.T)  
print("Correlation Matrix:")  
print(corr_matrix)  
 
# Make predictions on the test set  
y_pred_sgd_test = sgdr.predict(X_norm_test)  
y_pred_test = np.dot(X_norm_test, w_norm) + b_norm  
print(f"Prediction using np.dot() and sgdr.predict match: {(y_pred_test == y_pred_sgd_test).all()}")  
print(f"Prediction on test set:\n{y_pred_test[:4]}")  
print(f"Target values \n{y_test[:4]}")  
 
# Create a CSV writer  
with open('outputCLAD.csv', 'w', newline='') as csvfile:  
    writer = csv.writer(csvfile)  
    # Write the header row  
    writer.writerow(['Predicted Values', 'Target Values'])  
    # Write the data rows  
    for pred, target in zip(y_pred_test, y_test):  
        writer.writerow([pred, target])  
 
# Function to take user input for prediction  
def predict_user_input():  
    user_input = []  
    for feature in X_features:  
        value = float(input(f"Enter value for {feature}: "))  
        user_input.append(value)  
     
    user_input = np.array(user_input).reshape(1, -1)  
    user_input_norm = scaler.transform(user_input)  
    prediction = sgdr.predict(user_input_norm)  
    print(f"Predicted Last Selling Price: {prediction[0]}")  
 
# Call the function to predict based on user input  
predict_user_input()  