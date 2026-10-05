# 🛍️ E-Commerce Customer Retention & Churn Intelligence Pipeline

An end-to-end Data Analytics and Machine Learning project that transforms raw retail transaction logs into an actionable predictive churn dashboard. This project covers database normalization, advanced SQL feature engineering, scikit-learn classification modeling, and an interactive Streamlit web application.

---

## 🚀 Project Overview & Architecture

1. **Data Wrangling & PostgreSQL Normalization (Phase 1):**
   - Cleaned over 540,000 raw transaction records (`OnlineRetail.csv`).
   - Handled missing customer IDs, return anomalies, and character encodings.
   - Structured the dataset into normalized relational tables (`customers`, `products`, `transactions`).

2. **Advanced SQL & RFM Analysis (Phase 2):**
   - Extracted Recency, Frequency, and Monetary (RFM) behavioral metrics per customer using advanced PostgreSQL queries.
   - Built a clean analytical view (`customer_rfm`) to power downstream modeling.

3. **Machine Learning Classification (Phase 3):**
   - Engineered a binary target variable for 30-day customer churn based on a 90-day inactivity threshold.
   - Trained a scikit-learn classification model achieving **99.1% accuracy** (0.99 F1-score).
   - Serialized and saved the trained model and feature scaler using `joblib` (`churn_model.pkl`, `scaler.pkl`).

4. **Interactive Dashboard (`Streamlit`) (Phase 4):**
   - Developed an executive web application featuring high-level portfolio KPIs, dynamic international country filters, and real-time customer churn risk scoring.
