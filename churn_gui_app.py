import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")

# Load and train model
@st.cache_data
def load_and_train():
    df = pd.read_excel(r'E:\ITI(BI) - 4Months\Data Mining\Lab\Lab.02\churn_dataset.xlsx')
    df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

    x = df[['Age', 'Tenure', 'Sex']]
    y = df['Churn']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)
    model = GaussianNB()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    return model, acc, df

model, acc, df = load_and_train()

st.sidebar.header("🧾 Input Customer Info")
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (Months)", 0, 120, 12)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

input_data = np.array([[age, tenure, sex_val]])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

st.sidebar.markdown("---")
st.sidebar.metric("Model Accuracy", f"{acc*100:.2f}%")

with st.expander("📄 Preview Dataset"):
    st.dataframe(df)
