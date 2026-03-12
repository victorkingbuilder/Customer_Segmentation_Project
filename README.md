# Customer Segmentation Project

An end-to-end customer segmentation project using **K-Means clustering**, **feature scaling**, and **Streamlit** to classify customers based on demographic and purchasing behavior.

## Overview

This project applies unsupervised machine learning to group customers into meaningful segments using behavioral and spending-related features. It includes data preprocessing, model training, saved machine learning artifacts, and a Streamlit web app for predicting the segment of a new customer.

The goal is to help businesses better understand customer patterns and support targeted marketing, personalization, and decision-making.

## Features

* Customer segmentation using **K-Means Clustering**
* Data preprocessing with **Pandas**
* Feature scaling using **StandardScaler**
* Model serialization with **Joblib**
* Interactive prediction interface built with **Streamlit**
* Clean deployment-ready Python workflow

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib / Seaborn *(if used in exploration)*

## Project Structure

```bash
Customer_Segmentation_Project/
│
├── Analysis_Model.ipynb                     # Data analysis, preprocessing, training, evaluation
├── Segmentation.py                          # Streamlit app for customer segment prediction
├── customer_segmentation_kmeans_model.pkl   # Trained K-Means model
├── customer_segmentation_scaler.pkl         # Saved scaler
├── requirements.txt                         # Project dependencies
└── README.md                                # Project documentation
```

## Dataset Features Used

The trained model uses the following input features:

* Age
* Income
* Total_Spending
* NumWebPurchases
* NumStorePurchases
* NumWebVisitsMonth
* Recency

These exact feature names must be preserved during prediction to match the trained scaler and clustering model.

## How It Works

### 1. Data Preparation

The dataset is cleaned and selected customer-related numerical features are prepared for clustering.

### 2. Feature Scaling

Since clustering is distance-based, the features are standardized using `StandardScaler`.

### 3. Clustering

A **K-Means** model is trained on the scaled customer data to discover hidden customer groups.

### 4. Deployment

The trained scaler and K-Means model are saved and loaded into a **Streamlit** app where users can enter customer values and get a predicted segment.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-segmentation-project.git
cd customer-segmentation-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run Segmentation.py
```

## Example Usage

Enter customer information such as:

* Age
* Annual Income
* Total Spending
* Number of Web Purchases
* Number of Store Purchases
* Number of Web Visits
* Recency

Then click **Predict Segment** to see the assigned customer cluster.

## Important Note

The Streamlit app must send prediction data using the exact feature names used during model training:

```python
["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
```

Using different labels like `"Annual Income (k$)"` or `"Number of Web Purchases"` directly in the prediction DataFrame will cause a feature-name mismatch error with the scaler.

## Sample Requirements

```txt
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

## Future Improvements

* Add cluster interpretation and naming
* Visualize customer segments in the Streamlit app
* Deploy the app online using Streamlit Community Cloud
* Add model evaluation and elbow-method visualization
* Include customer recommendations based on predicted segment

## Use Cases

* Customer behavior analysis
* Marketing campaign targeting
* Personalized offers and promotions
* Customer retention strategies
* Business intelligence and segmentation insights

## Author

**Victor Nkwocha**

## License

This project is open for learning, academic, and portfolio purposes.

---

Here is a shorter, more polished **README intro section** you can place at the very top if you want it to feel more professional:

```md
# Customer Segmentation Project

This project builds a customer segmentation system using K-Means clustering and deploys it through a Streamlit web application. It analyzes customer demographics and purchasing behavior to group customers into meaningful segments that can support smarter marketing and business decisions.
```

And here is a strong `requirements.txt` you can use:

```txt
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.3,<2.0
scikit-learn>=1.2.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
