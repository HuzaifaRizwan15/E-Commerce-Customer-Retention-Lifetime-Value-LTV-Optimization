import streamlit as pd
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

st.set_page_config(
    page_title="E-Commerce Churn & LTV Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_data_and_models():
    
    db_user = 'postgres'
    db_password = ''  
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'ecommerce_churn_db'
    
    connection_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_string)
    
    df = pd.read_sql("SELECT * FROM customer_rfm;", con=engine)
    
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    return df, model, scaler

try:
    df_rfm, model, scaler = load_data_and_models()
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

st.sidebar.title("Navigation & Filters")
page = st.sidebar.selectbox("Select View", ["Executive Overview", "Customer Churn Risk Scoring"])

selected_country = st.sidebar.selectbox("Filter by Country", ["All"] + list(df_rfm['country'].unique()))
if selected_country != "All":
    filtered_df = df_rfm[df_rfm['country'] == selected_country]
else:
    filtered_df = df_rfm

if page == "Executive Overview":
    st.title("E-Commerce Customer Retention & LTV Dashboard")
    

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(filtered_df):,}")
    col2.metric("Avg Recency (Days)", f"{filtered_df['recency'].mean():.1f}")
    col3.metric("Avg Frequency", f"{filtered_df['frequency'].mean():.1f}")
    col4.metric("Total Monetary Value", f"${filtered_df['monetary'].sum():,.2f}")
    
    st.markdown("---")
    
    st.subheader("Top 10 High-Value Customers by Monetary Spend")
    st.dataframe(filtered_df.sort_values(by='monetary', ascending=False).head(10))

elif page == "Customer Churn Risk Scoring":
    st.title("30-Day Customer Churn Risk Prediction")
   
    X_pred = filtered_df[['recency', 'frequency', 'monetary']]
    X_scaled = scaler.transform(X_pred)
    filtered_df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
    filtered_df['predicted_churn'] = model.predict(X_scaled)
   
    churned_count = filtered_df['predicted_churn'].sum()
    total_count = len(filtered_df)
    churn_rate = (churned_count / total_count) * 100 if total_count > 0 else 0
    
    col1, col2 = st.columns(2)
    col1.metric("At-Risk Customers (Churned)", f"{churned_count:,}")
    col2.metric("Portfolio Churn Rate", f"{churn_rate:.2f}%")
    
    st.markdown("---")
    st.subheader("Customer Accounts Flagged for Immediate Retention Action")
    
    risk_df = filtered_df.sort_values(by='churn_probability', ascending=False)[
        ['customer_id', 'country', 'recency', 'frequency', 'monetary', 'churn_probability']
    ]
    st.dataframe(risk_df.head(25))