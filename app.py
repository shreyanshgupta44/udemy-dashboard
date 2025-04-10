import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import base64

st.set_page_config(page_title="Udemy Course Dashboard", layout="wide")

# Title
st.title("📚 Udemy Course Success Predictor & Visualizer")

# Load dataset
df = pd.read_csv("udemy_courses.csv")

# Data Prep
df['is_successful'] = df['num_subscribers'].apply(lambda x: 1 if x > 10000 else 0)
le = LabelEncoder()
df['level_encoded'] = le.fit_transform(df['level'])
df['subject_encoded'] = le.fit_transform(df['subject'])

# Train Model
X = df[['price', 'num_lectures', 'content_duration', 'is_paid', 'level_encoded', 'subject_encoded']]
y = df['is_successful']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# Columns layout
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("📍 Precision", f"{precision * 100:.2f}%")
col3.metric("🚀 Recall", f"{recall * 100:.2f}%")
col4.metric("💡 F1 Score", f"{f1 * 100:.2f}%")
col5.metric("📊 AUC", f"{auc:.2f}")

# Sidebar
st.sidebar.title("📁 Visualizations")
option = st.sidebar.selectbox("Select a graph", [
    "Feature Importance", "Subject vs Success", "Level Distribution", "Price Distribution"
])

# Visualization
if option == "Feature Importance":
    st.subheader("📌 Feature Importance from Random Forest")
    importances = model.feature_importances_
    features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    fig, ax = plt.subplots()
    sns.barplot(data=features_df, x='Importance', y='Feature', ax=ax, palette="viridis")
    st.pyplot(fig)

elif option == "Subject vs Success":
    st.subheader("📘 Subject vs Success Rate")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='subject', hue='is_successful', ax=ax, palette='pastel')
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif option == "Level Distribution":
    st.subheader("📚 Course Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='level', ax=ax)
    st.pyplot(fig)

elif option == "Price Distribution":
    st.subheader("💰 Price Distribution (Under ₹400)")
    fig, ax = plt.subplots()
    sns.histplot(data=df[df['price'] < 400], x='price', bins=30, kde=True, ax=ax, color='purple')
    st.pyplot(fig)

# Live Prediction
st.markdown("## 🎯 Try Live Prediction")

with st.form("predict_form"):
    price = st.number_input("Course Price", min_value=0, max_value=500, value=100)
    num_lectures = st.number_input("Number of Lectures", min_value=1, max_value=500, value=20)
    duration = st.number_input("Content Duration (hours)", min_value=0.0, max_value=100.0, value=5.0)
    is_paid = st.selectbox("Is the course Paid?", ["Yes", "No"])
    level = st.selectbox("Course Level", df['level'].unique())
    subject = st.selectbox("Subject", df['subject'].unique())
    submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame({
            'price': [price],
            'num_lectures': [num_lectures],
            'content_duration': [duration],
            'is_paid': [1 if is_paid == "Yes" else 0],
            'level_encoded': [le.fit(df['level']).transform([level])[0]],
            'subject_encoded': [le.fit(df['subject']).transform([subject])[0]]
        })

        prediction = model.predict(input_data)[0]
        st.success("✅ This course is likely to be successful!" if prediction else "⚠️ This course may not perform well.")

# Download processed data
st.markdown("### 📥 Download Processed Data")
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df)
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name='processed_udemy_data.csv',
    mime='text/csv',
)
