# AI2018L_CREDIT-CARD-FRAUD-DETECTION_2021
Credit Card Fraud Detection is an Artificial Intelligence Project to detect fraudulent or non-fraudulent credit card transactions by applying suitable AI / ML Algorithms on Kaggle Credit Card Fraud Detection Dataset.

Link to Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

Environment used for deployment: Flask

The main website file: app.py

The file where model has been trained: main.py

The file in which different graphs of data has been plotted: graphs.py

web pages present in template folder i.e., home.html and result.html

stylesheet is present in static folder i.e., index.css

pickle library has been used to convert the model in main.py file into model.pkl which can be easily loaded into flask  application

Working of Website: The Website will take 30 input features from users as dataset consists of 30 features(columns) and then will predict that whether the transaction is fraudulent or not after applying the trained ML model on the values provided by the user.


Data in Credit Card Fraud Detection Dataset of Kaggle is highly imbalanced as only 492 transactions are fraudulent out of 284807 transactions. Our team has used Logistic regression with balanced class , tomeklinks and SMOTE techniques to resolve this issue and to have a prediction upto an accurate level.


