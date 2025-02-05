# Customer-Propensity
This code performs several tasks to analyze and predict customer purchase behavior using machine learning. Here's a detailed description of its components and flow:

### 1. **Importing Libraries**
   - Various Python libraries are used for data manipulation, model training, and evaluation. These include:
     - **matplotlib & seaborn** for visualizations.
     - **pandas** for data manipulation.
     - **scikit-learn** for machine learning algorithms and evaluation.
  
### 2. **Loading the Data**
   - The data is loaded from a CSV file, located at `'C:\Users\amaan\Desktop\c2\customerpropensity2.csv'`.
   - The target variable is `ordered` (indicating whether a customer has placed an order or not), and features include user interaction data such as clicks and device type.

### 3. **Data Exploration**
   - Basic exploration of the dataset is performed using `df.head()`, `df.info()`, `df.describe()`, and `df.isnull().sum()` to understand the structure, check for missing values, and summarize statistics.
   - A count plot is used to visualize the distribution of the target variable (`ordered`).

### 4. **Feature Selection**
   - The correlation heatmap is plotted to identify relationships between features.
   - A custom function `correlation()` is used to identify and drop highly correlated features (with a threshold of 0.7) to avoid multicollinearity.

### 5. **Data Preprocessing**
   - The features (`X`) and target (`y`) are prepared by separating the `ordered` column as the target.
   - The data is split into training and test sets (70% train, 30% test).
   - Standard scaling is applied to normalize the feature values, ensuring that they are on the same scale for better model performance.

### 6. **Model Training and Evaluation**
   - **Logistic Regression** and **Bernoulli Naive Bayes** models are trained on the scaled training data.
   - The models' performance is evaluated using accuracy, confusion matrix, and classification report (precision, recall, F1-score) on the test data.

### 7. **Prediction Function**
   - The function `predict_customer_purchase()` allows for predicting whether a customer is likely to place an order based on input values (such as user interactions like clicks, sign-ins, and device type).
   - The user input is processed to match the training data structure, and predictions are made using the trained Bernoulli Naive Bayes model.

### 8. **Test Cases for Prediction**
   - Several test cases are defined with different combinations of user interaction features to simulate customer behavior.
   - The predictions for these test cases are printed, indicating whether the customer is likely to place an order (`'Ordered'` or `'Not Ordered'`).

### 9. **Final Model Predictions**
   - The model (Bernoulli Naive Bayes) is retrained on the full dataset, and the prediction function is called on the defined test cases.
   - The results are displayed for each test case, predicting whether the customer is likely to order based on their behavior.

### Key Points:
- **Data Preparation**: The code performs various preprocessing steps like scaling features and handling correlations.
- **Modeling**: It trains multiple models (Logistic Regression and Naive Bayes) and evaluates their performance.
- **Customer Prediction**: The code includes a method to predict whether a customer will order based on input features.
