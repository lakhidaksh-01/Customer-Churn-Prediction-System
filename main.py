import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
import joblib

# =======================
# 🎯 Load Model & Data
# =======================
model = joblib.load("model.pkl")  # trained model
x_train = pd.read_csv("x_train.csv")
x_test = pd.read_csv("x_test.csv")
id_test = pd.read_csv("id_test.csv")


id_test = id_test.squeeze()  # Convert from DataFrame → Series
# =======================
# 🧭 Page Configuration
# =======================
st.set_page_config(
    page_title="Telco Churn Predictor — Explainable AI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =======================
# 🎨 Modern Professional Styling
# =======================
st.markdown("""
    <style>
        /* Base Layout */
        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
            background-color: #f7f9fc;
            color: #1f2937;
        }

        .main {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.08);
        }

        /* Title */
        h1 {
            text-align: center;
            font-weight: 700;
            color: #1e293b;
            font-size: 2.2rem;
        }

        h3 {
            color: #334155;
            font-weight: 600;
        }

        /* Metric Cards */
        .metric-box {
            background: linear-gradient(145deg, #ffffff, #f3f6fa);
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-box:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        }
        .metric-box h2 {
            margin: 0;
            color: #0284c7;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .metric-box p {
            margin: 0;
            font-size: 15px;
            color: #475569;
        }

        /* Inputs */
        .stNumberInput > div > div {
            border-radius: 10px !important;
            border: 1px solid #cbd5e1 !important;
            background-color: #f9fafb !important;
        }

        /* Section Headings */
        .stMarkdown h3 {
            border-left: 4px solid #0284c7;
            padding-left: 10px;
            color: #1e293b !important;
            margin-top: 2rem;
        }

        /* Footer */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =======================
# 🏷️ Header
# =======================
st.title("📊 Telco Churn Predictor — Explainable AI Dashboard")
st.markdown(
    "<p style='text-align:center; font-size:18px; color:#475569;'>Predict customer churn, visualize model insights, and explore interpretable AI explanations for decision-making.</p>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

# =======================
# 🧍‍♂️ Customer Selection Inline
# =======================
col1, col2 = st.columns([1, 4])
with col1:
    customer_id = st.selectbox("Select Customer ID",id_test)
    customer_index = id_test[customer_id == id_test].index[0]
    customer = x_test.iloc[[customer_index]]

with col2:
    st.markdown(
        "<p style='margin-top:18px; font-size:16px; color:#475569;'>Select a customer index to generate predictions and explanations.</p>",
        unsafe_allow_html=True
    )

# =======================
# 🔮 Prediction Results
# =======================
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]

st.markdown("### 🎯 Prediction Results")
col3, col4 = st.columns(2)
with col3:
    st.markdown("<div class='metric-box'><h2>{}</h2><p>Prediction</p></div>".format(
        "🚨 Churn" if prediction == 1 else "✅ Not Churn"), unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-box'><h2>{:.2f}</h2><p>Churn Probability</p></div>".format(
        probability*100), unsafe_allow_html=True)

# =======================
# 📊 Feature Importance
# =======================
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=x_train.columns).sort_values(ascending=True)

fig1, ax = plt.subplots(figsize=(6, 4))
feat_imp[-10:].plot(kind='barh', color='#38bdf8', ax=ax)
ax.set_title("Top 10 Features Influencing Churn", fontsize=12, fontweight='bold', color='#1e293b')
ax.set_xlabel("Importance Score", fontsize=10, color='#334155')
ax.set_ylabel("Feature", fontsize=10, color='#334155')
ax.tick_params(colors='#334155')
ax.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()

# =======================
# 🧠 LIME Explanation
# =======================
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_train.columns,
    class_names=['Not Churn', 'Churn'],
    mode="classification"
)
exp = explainer.explain_instance(
    data_row=customer.values[0],
    predict_fn=model.predict_proba
)

fig2 = exp.as_pyplot_figure(label=1)
plt.title("LIME Explanation: Feature Influence on Churn Probability", fontsize=12, fontweight='bold', color='#1e293b')
plt.xlabel("Impact on Churn Prediction", fontsize=10, color='#334155')
plt.ylabel("Feature", fontsize=10, color='#334155')
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

# =======================
# 📈 Display both charts in one row
# =======================
col5, col6 = st.columns(2)
with col5:
    st.pyplot(fig2, use_container_width=True)
with col6:
    st.pyplot(fig1, use_container_width=True)

# =======================
# 📂 Customer Data Explorer
# =======================
st.markdown("### 📂 Customer Data Explorer")
st.markdown(
    "<p style='color:#5f6b73;'>Explore the original Telco customer data to find specific indexes for prediction.</p>",
    unsafe_allow_html=True
)
with st.expander("🔍 View and Explore Customer Data", expanded=False):
    df_preview = x_test.copy()
    df_preview.insert(0, "customerID", id_test.values)
    st.dataframe(df_preview.reset_index(drop=True), use_container_width=True, height=350)

# =======================
# 🧾 Footer
# =======================
st.markdown("<hr style='border: 1px solid #e2e8f0; margin-top:40px;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#64748b;'>Built with ❤️ by <b>Daksh Lakhi</b> — Explainable AI in action ⚡</p>",
    unsafe_allow_html=True
)