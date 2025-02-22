##Project Description: Customer Churn Prediction with Segmentation and Model Optimization

This project focuses on predicting customer churn by analyzing various customer attributes and using machine learning techniques, including data preprocessing, customer segmentation, and model optimization. The pipeline also incorporates MLflow for experiment tracking and model management. Below is a detailed description of the key steps involved:

1. Data Preparation and Preprocessing:
Loading the Dataset: The dataset Churn_Modelling.csv is loaded, containing customer-related information such as demographics, account details, and whether the customer exited or not.

Label Encoding and One-Hot Encoding:

Gender Encoding: The categorical Gender column is transformed into numerical values using LabelEncoder, where 'Male' is encoded as 1 and 'Female' as 0.
Geography Encoding: The Geography column is encoded using OneHotEncoder, creating binary columns for each geographic region (France, Germany, Spain).
Data Cleaning:

The dataset is cleaned by removing unnecessary columns such as Surname and RowNumber, and by handling missing values (dropna).
2. Customer Segmentation:
Standardization: The numerical features are standardized using StandardScaler, which normalizes the feature scales and ensures all features are treated equally during clustering.

K-Means Clustering:

The Elbow Method is used to identify the optimal number of clusters by plotting the inertia (sum of squared distances from the cluster centers).
Based on the elbow plot, 3 clusters are chosen for segmenting customers into distinct groups.
Dimensionality Reduction:

PCA (Principal Component Analysis) is applied to reduce the dimensionality of the dataset to 2D, facilitating visualization of the customer segments.
The resulting 2D plot shows how customers are grouped into clusters based on their features.
Cluster Analysis:

After clustering, the data is grouped by the Cluster label, and the mean of each feature is calculated to analyze the characteristics of each customer group. These segments can help in identifying customer patterns and behaviors.
3. Model Building:
Random Forest Classifier:

RandomForestClassifier is used to predict whether a customer will churn (Exited). The model is a robust ensemble method that works well with structured data.
Hyperparameter Tuning with GridSearchCV:

GridSearchCV is utilized to tune the hyperparameters of the Random Forest model, including n_estimators, max_depth, min_samples_split, and min_samples_leaf. This improves model performance by selecting the best combination of hyperparameters through cross-validation.
Train-Test Split:

The dataset is split into training and test sets using train_test_split to prevent data leakage and ensure proper model evaluation.
4. Handling Imbalanced Data:
SMOTE (Synthetic Minority Over-sampling Technique):
SMOTE is applied to balance the class distribution in the training set, particularly addressing the problem of class imbalance in the Exited column. SMOTE creates synthetic samples for the minority class (non-churn customers) to improve model performance.
5. Model Training and Evaluation:
Model Training:

The optimized Random Forest model is trained on the resampled training data, and the best performing model is selected through GridSearchCV.
Model Evaluation:

Model performance is assessed using several evaluation metrics: accuracy, precision, recall, and f1-score. These metrics help gauge the model's ability to correctly predict churn versus non-churn customers.
MLflow Integration:

MLflow is used to track and log experiment details, including model parameters, metrics, and the final model.
Experiment Tracking: MLflow logs each experiment's parameters (such as the best hyperparameters) and metrics (such as precision, recall, and F1 score).
Model Registration: The best model is logged into MLflow as a registered model (Churn_RF_Model), making it easy to deploy and monitor.
6. Conclusion and Insights:
Customer Segments: By using K-Means clustering, customers are segmented into three distinct groups, each with different features such as account balance and geography.

Cluster 0: Customers with higher account balances and a moderate likelihood of churning.
Cluster 1: Customers with lower balances and a lower likelihood of churning.
Cluster 2: Customers with medium balances and a moderate to low likelihood of exiting.
Targeted Marketing and Retention: These customer segments provide valuable insights for targeting specific customer groups with personalized retention strategies.

Key Technologies and Libraries Used:
Data Preprocessing: pandas, scikit-learn, imbalanced-learn
Machine Learning: RandomForestClassifier, GridSearchCV, SMOTE
Clustering and Segmentation: KMeans, PCA
Model Experimentation and Tracking: MLflow
Visualization: matplotlib
