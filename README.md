Scrap Price Prediction Using SGD Regression

This project aims to predict the selling price of scrap materials based on several key features using Stochastic Gradient Descent (SGD) Regression. The model is trained and tested using a dataset of scrap prices, and it offers predictions based on user inputs.

Project Overview

Scrap price prediction is crucial for industries dealing with recycled materials, as it helps in determining the future market trends and optimizing procurement strategies. This project uses a regression model to forecast scrap prices based on factors such as:


Price with Overhead Cost Per MT

Grade

Alang Scrap Price

Fresh Procurement Price


The model has been developed using Python and leverages the following libraries:


numpy

pandas

scikit-learn

Dataset

The dataset used in this project contains historical data on scrap prices, including the above-mentioned features. The data is preprocessed to handle missing values, and feature scaling is applied to normalize the input data for better performance of the SGD regressor.



Model Training and Evaluation

The project employs the SGDRegressor from scikit-learn for model training. The data is split into training and testing sets, with the model being trained on 80% of the data and evaluated on the remaining 20%. Key performance metrics such as Mean Squared Error (MSE) and R-squared (R2) are calculated to assess the model's accuracy.



Features and Functionality

Model Training: The model is trained using the SGD algorithm, which is efficient for large-scale and sparse datasets.

Prediction: The trained model can predict the last selling price of scrap materials based on new inputs provided by the user.

User Interaction: A function is included to allow users to input their own feature values and receive a predicted scrap price.

Getting Started

To run the project, ensure you have the required Python packages installed. You can then execute the script to train the model, evaluate its performance, and make predictions based on user input.


Usage

Clone the repository:

git clone https://github.com/harshiv03/scrap-price-prediction.git


Install dependencies:

pip install -r requirements.txt


Run the script:

python scrap_price_prediction.py


Input your data:

Follow the on-screen prompts to input the feature values and get the predicted scrap price.

Results

The model's predictions are written to a CSV file, and the accuracy metrics are displayed in the console. 
The correlation matrix of the features is also provided to give insights into the relationships between different variables.

Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
