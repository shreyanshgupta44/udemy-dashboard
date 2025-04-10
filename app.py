import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv('udemy_courses.csv')

# Preprocessing
df['is_successful'] = df['num_subscribers'].apply(lambda x: 1 if x > 10000 else 0)
le = LabelEncoder()
df['level_encoded'] = le.fit_transform(df['level'])
df['subject_encoded'] = le.fit_transform(df['subject'])

# Sidebar options
st.sidebar.title("Filters")
show_graph = st.sidebar.selectbox("Choose a Visualization", [
    "None", "Course Levels", "Success by Subject", "Price Distribution"
])

# Main title
st.title("Udemy Course Success Dashboard")

# Metrics section
X = df[['price', 'num_lectures', 'content_duration', 'is_paid', 'level_encoded', 'subject_encoded']]
y = df['is_successful']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.subheader("Model Evaluation")
st.metric("Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
st.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.2f}")

# Visualizations
if show_graph == "Course Levels":
    st.subheader("Course Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='level', ax=ax)
    st.pyplot(fig)

elif show_graph == "Success by Subject":
    st.subheader("Successful vs Unsuccessful Courses by Subject")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='subject', hue='is_successful', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif show_graph == "Price Distribution":
    st.subheader("Price Distribution (Under ₹400)")
    fig, ax = plt.subplots()
    sns.histplot(data=df[df['price'] < 400], x='price', bins=30, ax=ax)
    st.pyplot(fig)
