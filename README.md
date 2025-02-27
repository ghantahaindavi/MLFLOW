# Customer Churn Prediction with Segmentation and Model Optimization

This project focuses on predicting customer churn by analyzing various customer attributes and using machine learning techniques, including data preprocessing, customer segmentation, and model optimization. The pipeline also incorporates **MLflow** for experiment tracking and model management.

---

## 📌 Project Workflow

### 1️⃣ Data Preparation and Preprocessing
- **Loading the Dataset:** The dataset (`Churn_Modelling.csv`) contains customer-related information such as demographics, account details, and churn status.
- **Label Encoding and One-Hot Encoding:**
  - `Gender Encoding`: Converted categorical `Gender` into numerical values (`Male = 1, Female = 0`) using `LabelEncoder`.
  - `Geography Encoding`: Used `OneHotEncoder` to create binary columns for geographic regions (`France, Germany, Spain`).
- **Data Cleaning:**
  - Removed unnecessary columns (`Surname, RowNumber`) and handled missing values (`dropna`).

---

### 2️⃣ Customer Segmentation
- **Standardization:** Applied `StandardScaler` to normalize numerical features for effective clustering.
- **K-Means Clustering:**
  - Used the **Elbow Method** to determine the optimal number of clusters.
  - Selected **3 clusters** to segment customers.
- **Dimensionality Reduction:**
  - Applied **PCA (Principal Component Analysis)** to reduce data to **2D** for better visualization.
  - Plotted **customer clusters** based on key attributes.
- **Cluster Analysis:** Analyzed customer segments based on feature distributions.

---

### 3️⃣ Model Building
- **Random Forest Classifier:**
  - Implemented `RandomForestClassifier` for churn prediction.
- **Hyperparameter Tuning:**
  - Used `GridSearchCV` to optimize parameters (`n_estimators, max_depth, min_samples_split, min_samples_leaf`).
- **Train-Test Split:** Used `train_test_split` to split data into training and test sets.

---

### 4️⃣ Handling Imbalanced Data
- **SMOTE (Synthetic Minority Over-sampling Technique):**
  - Applied `SMOTE` to **balance the dataset**, generating synthetic samples for the minority class.

---

### 5️⃣ Model Training and Evaluation
- **Training:** Optimized **Random Forest model** trained on balanced data.
- **Evaluation Metrics:** Assessed using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- **MLflow Integration:**
  - **Experiment Tracking:** Logged model parameters and metrics using `MLflow`.
  - **Model Registration:** Stored the best-performing model as `Churn_RF_Model` for future deployment.

---

### 6️⃣ Conclusion and Insights
- **Customer Segments Identified:**
  - **Cluster 0:** High account balance, moderate churn risk.
  - **Cluster 1:** Low balance, low churn risk.
  - **Cluster 2:** Medium balance, moderate churn risk.
- **Business Impact:**
  - Insights help businesses **target customer groups** with personalized retention strategies.

---

## 🔧 Key Technologies and Libraries Used
| Category               | Libraries/Tools |
|------------------------|----------------|
| **Data Preprocessing** | `pandas, scikit-learn, imbalanced-learn` |
| **Machine Learning**   | `RandomForestClassifier, GridSearchCV, SMOTE` |
| **Clustering**         | `KMeans, PCA` |
| **Model Experimentation** | `MLflow` |
| **Visualization**      | `matplotlib` |

---

## 📂 Project Structure
