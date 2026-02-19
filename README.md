# 📉 Telco Customer Churn Prediction App (XGBoost + Streamlit + AWS EC2)

This project demonstrates a complete end-to-end Machine Learning deployment workflow — from data preprocessing and model training to building a frontend and deploying the application on AWS EC2.

It combines:

* Feature engineering
* SMOTE for imbalance handling
* XGBoost with hyperparameter tuning
* Streamlit frontend
* Cloud deployment using Ubuntu EC2

---

# 🚀 Project Overview

The goal of this project is to predict whether a telecom customer is likely to churn based on:

* Demographics
* Services subscribed
* Contract type
* Billing information
* Tenure category
* Monthly and total charges

The model is trained using:

* XGBoost Classifier
* SMOTE for class imbalance
* Stratified Cross Validation
* F1-score optimization

---

# 🧠 Machine Learning Pipeline

The saved model file (`churn_xgb_smote_pipeline.pkl`) contains a full pipeline consisting of:

1. ColumnTransformer

   * StandardScaler for numerical features
   * OneHotEncoder for categorical features
2. SMOTE (applied only during training)
3. Tuned XGBoost Classifier

This ensures:

* Consistent preprocessing
* No manual feature alignment during inference
* Safe handling of unseen categories

---

# 📊 Feature Engineering

The following preprocessing steps were applied:

* Removed `customerID`
* Cleaned and converted `TotalCharges` to numeric
* Filled missing `TotalCharges` with median
* Converted `Churn` to binary (0/1)
* Converted `tenure` into categorical bins:

  * 0–12
  * 13–24
  * 25–36
  * 37–48
  * 49–60
  * 61–72
* Dropped original `tenure` column

---

# 🖥️ Streamlit Frontend

The Streamlit application allows users to:

* Enter customer details through a structured form
* Submit inputs
* Get churn prediction
* View churn probability

Run locally:

```
streamlit run app.py
```

---

# ☁️ AWS EC2 Deployment

Deployment was done using:

* Ubuntu EC2 instance (t3.micro – Free Tier eligible)
* Security Group allowing:

  * Port 22 (SSH)
  * Port 8501 (Streamlit)
* Virtual environment for dependency isolation

Deployment Steps:

1. Launch EC2 instance
2. SSH into server
3. Clone GitHub repository
4. Create virtual environment
5. Install dependencies
6. Run Streamlit

Start the server:

```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Access the app:

```
http://PUBLIC_IP:8501
```

---

# 🔄 Accessing After Stopping EC2

If the EC2 instance is stopped:

1. Start the instance from AWS Console
2. Wait until status checks pass
3. Copy the new Public IPv4 address (may change)
4. SSH into server
5. Activate virtual environment
6. Run Streamlit again

Stopping the instance shuts down the app.

---

# 🔐 Understanding the Endpoint

Current endpoint format:

```
http://PUBLIC_IP:8501
```

Architecture:

User Browser
↓
Public Internet
↓
AWS EC2 Public IP
↓
Port 8501
↓
Streamlit Server
↓
XGBoost Pipeline
↓
Prediction Response

The application runs entirely inside the EC2 instance. Once deployed, it does not depend on the local machine.

---

# 📦 Requirements

```
streamlit==1.32.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
imbalanced-learn==0.12.3
xgboost==2.1.4
joblib==1.4.2
matplotlib==3.9.0
```

---

# 🔄 Updating the Deployed App

If code is updated and pushed to GitHub:

1. SSH into EC2
2. Navigate to project directory
3. Run:

```
git pull
```

4. Restart Streamlit

---

# ⚠️ Important Notes

* Do not leave EC2 running unnecessarily to avoid charges.
* Public IP may change after restarting unless using Elastic IP.
* Streamlit development server is suitable for demos and portfolio projects.
* For production systems, use Nginx + HTTPS + domain.

---

# 🎯 Future Improvements

* Add Nginx reverse proxy
* Add HTTPS (SSL certificate)
* Connect custom domain
* Dockerize the application
* Implement CI/CD for auto-deployment

---

# 👨‍💻 Author
Hansraj Singh
End-to-end Machine Learning deployment project demonstrating practical model serving using cloud infrastructure and modern ML engineering workflow.
