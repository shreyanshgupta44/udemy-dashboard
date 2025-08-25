import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import base64

# --- Page Configuration ---
st.set_page_config(page_title="Online Customer Retention Prediction", layout="wide")

# --- Initialize Session State for Feedback ---
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# --- Title ---
st.title("📊 Online Customer Retention Prediction & Visualizer")

# --- Load Processed Dataset ---
@st.cache_data
def load_data():
    try:
        # NOTE: The underlying data is still from the Udemy dataset.
        # The column names are interpreted for a customer retention context.
        df = pd.read_csv("processed_udemy_data.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'processed_udemy_data.csv' not found. Please make sure the file is in the same directory.")
        return None

df = load_data()
if df is None:
    st.stop()

# --- Re-initialize LabelEncoders for Prediction Form ---
level_le = LabelEncoder()
df['level_encoded'] = level_le.fit_transform(df['level'])

subject_le = LabelEncoder()
df['subject_encoded'] = subject_le.fit_transform(df['subject'])


# --- Model Training ---
# The features are mapped to customer retention concepts for the UI.
# 'price' -> Service Cost, 'num_lectures' -> Interactions, 'content_duration' -> Usage
X = df[['price', 'num_lectures', 'content_duration', 'is_paid', 'level_encoded', 'subject_encoded']]
# 'is_successful' is re-interpreted as 'is_retained'
y = df['is_successful']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# --- Performance Metrics ---
st.markdown("### 📈 Model Performance Metrics")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("📍 Precision", f"{precision * 100:.2f}%")
col3.metric("🚀 Recall", f"{recall * 100:.2f}%")
col4.metric("💡 F1 Score", f"{f1 * 100:.2f}%")
col5.metric("📊 AUC", f"{auc:.2f}")


# --- Sidebar for Filters and Visualizations ---
st.sidebar.title("📁 Filters & Visualizations")

# Dynamic Filters
st.sidebar.markdown("### ⚙️ Dynamic Filters")
selected_segment = st.sidebar.selectbox("Filter by Customer Segment", ['All'] + list(df['subject'].unique()))
selected_tier = st.sidebar.selectbox("Filter by Service Tier", ['All'] + list(df['level'].unique()))

# Filter the dataframe
filtered_df = df.copy()
if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['subject'] == selected_segment]
if selected_tier != 'All':
    filtered_df = filtered_df[filtered_df['level'] == selected_tier]

option = st.sidebar.selectbox("Select a graph", [
    "Key Retention Drivers", "Segment vs Retention", "Service Tier Distribution", "Service Cost Distribution", "Customer Engagement (Reviews vs. Usage)"
])


# --- Main Panel for Visualizations ---
st.markdown("---")
st.subheader(f"📊 Visualization: {option}")

if not filtered_df.empty:
    if option == "Key Retention Drivers":
        st.write("Shows which customer features are most influential in predicting retention.")
        importances = model.feature_importances_
        # Renaming features for the chart
        feature_map = {'price': 'Service Cost', 'num_lectures': 'Interactions', 'content_duration': 'Service Usage', 'is_paid': 'On Paid Plan', 'level_encoded': 'Service Tier', 'subject_encoded': 'Customer Segment'}
        features_df = pd.DataFrame({'Feature': [feature_map.get(f, f) for f in X.columns], 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=features_df, x='Importance', y='Feature', ax=ax, palette="viridis")
        ax.set_title("Key Retention Drivers", fontsize=16)
        st.pyplot(fig)

    elif option == "Segment vs Retention":
        st.write("Compares the number of retained vs. churned customers for each segment.")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='subject', hue='is_successful', ax=ax, palette='pastel')
        plt.xticks(rotation=45, ha='right')
        ax.set_title("Customer Segment vs Retention Rate", fontsize=16)
        st.pyplot(fig)

    elif option == "Service Tier Distribution":
        st.write("Shows the distribution of customers across different service tiers.")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='level', ax=ax, palette="plasma")
        ax.set_title("Customer Service Tier Distribution", fontsize=16)
        st.pyplot(fig)

    elif option == "Service Cost Distribution":
        st.write("Displays the frequency of different service costs (for costs under 200).")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_df[filtered_df['price'] < 200], x='price', bins=30, kde=True, ax=ax, color='purple')
        ax.set_title("Service Cost Distribution (Under $200)", fontsize=16)
        st.pyplot(fig)

    elif option == "Customer Engagement (Reviews vs. Usage)":
        st.write("Shows the relationship between customer usage and reviews, colored by segment.")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=filtered_df, x='num_subscribers', y='num_reviews', hue='subject', ax=ax, alpha=0.7)
        ax.set_title("Customer Engagement: Reviews vs. Usage", fontsize=16)
        ax.set_xlabel("Usage Metric (e.g., Logins, Actions)")
        ax.set_ylabel("Number of Reviews")
        ax.set_xscale('log')
        ax.set_yscale('log')
        st.pyplot(fig)
else:
    st.warning("No data available for the selected filters.")

# --- High Retention Rate Customers ---
st.markdown("---")
st.markdown("## 🏆 Customers with High Predicted Retention Rate (>50%)")
with st.spinner('Analyzing customers...'):
    all_customers_proba = model.predict_proba(X)[:, 1]
    df['retention_probability'] = all_customers_proba
    high_retention_customers = df[df['retention_probability'] > 0.5].sort_values(by='retention_probability', ascending=False)
    high_retention_customers['retention_probability_percent'] = high_retention_customers['retention_probability'] * 100
    
    st.dataframe(high_retention_customers[['course_title', 'subject', 'price', 'num_subscribers', 'retention_probability_percent']].rename(columns={
        'course_title': 'Customer ID',
        'subject': 'Segment',
        'price': 'Service Cost ($)',
        'num_subscribers': 'Usage Metric',
        'retention_probability_percent': 'Retention Probability (%)'
    }))

# --- Live Prediction Form ---
st.markdown("---")
st.markdown("## 🔮 Predict Customer Retention")

with st.form("predict_form"):
    st.write("Enter the details of a hypothetical customer to predict their retention.")
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        price = st.number_input("Service Cost ($)", min_value=0, max_value=500, value=100)
        num_lectures = st.number_input("Interactions / Month", min_value=1, max_value=1000, value=30)
        duration = st.number_input("Service Usage (hours)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    with form_col2:
        is_paid_str = st.selectbox("Is Customer on a Paid Plan?", ["Yes", "No"])
        level = st.selectbox("Service Tier", options=level_le.classes_)
        subject = st.selectbox("Customer Segment", options=subject_le.classes_)
    submit = st.form_submit_button("Predict Retention")
    if submit:
        level_encoded_input = level_le.transform([level])[0]
        subject_encoded_input = subject_le.transform([subject])[0]
        is_paid_input = 1 if is_paid_str == "Yes" else 0
        input_data = pd.DataFrame({
            'price': [price], 'num_lectures': [num_lectures], 'content_duration': [duration],
            'is_paid': [is_paid_input], 'level_encoded': [level_encoded_input], 'subject_encoded': [subject_encoded_input]
        })
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.success(f"✅ This customer is likely to be **retained**! (Probability: {prediction_proba:.2%})")
        else:
            st.warning(f"⚠️ This customer may be at **risk of churning**. (Retention Probability: {prediction_proba:.2%})")

# --- Feedback Section ---
st.markdown("---")
st.markdown("## 📝 Customer Feedback")
customer_list = sorted(df['course_title'].unique())
selected_customer_for_feedback = st.selectbox("Select a Customer ID to leave feedback for:", customer_list)

feedback_text = st.text_area("Your feedback here...", key="feedback_input")
if st.button("Submit Feedback"):
    if feedback_text and selected_customer_for_feedback:
        if selected_customer_for_feedback not in st.session_state.feedback:
            st.session_state.feedback[selected_customer_for_feedback] = []
        st.session_state.feedback[selected_customer_for_feedback].append(feedback_text)
        st.success("Thank you for your feedback! It has been recorded.")
    else:
        st.warning("Please select a customer and enter some feedback before submitting.")

# --- Display Submitted Feedback ---
st.markdown("### 💬 Submitted Feedback")
if st.session_state.feedback:
    for customer, feedbacks in st.session_state.feedback.items():
        with st.expander(f"Feedback for Customer **{customer}** ({len(feedbacks)})"):
            for i, fb in enumerate(feedbacks):
                st.info(f"**Feedback #{i+1}:** {fb}")
else:
    st.write("No feedback has been submitted yet.")

# --- Download Processed Data ---
st.markdown("---")
st.markdown("### 📥 Download Data")
@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')
csv = convert_df_to_csv(df)
st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name='customer_data.csv',
    mime='text/csv',
)
