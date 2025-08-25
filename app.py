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
st.set_page_config(page_title="Customer Retention Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for a Professional UI ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Card-like containers for sections */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        border-radius: 15px;
        padding: 25px;
        background-color: white;
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.05);
        transition: 0.3s;
    }

    /* Style for metric cards */
    .metric-card {
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.05);
    }
    
    /* Ensure metric text is dark and readable */
    .metric-card div[data-testid="stMetricValue"], .metric-card div[data-testid="stMetricLabel"] {
        color: #333333 !important;
    }

    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #0072C6;
        background-color: #0072C6;
        color: white;
        padding: 10px 24px;
        transition-duration: 0.4s;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: white;
        color: #0072C6;
        border: 1px solid #0072C6;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State for Feedback ---
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# --- Header ---
with st.container():
    st.title("📊 Customer Retention Dashboard")
    st.markdown("An interactive dashboard to predict customer churn and analyze key retention drivers.")
    st.markdown("---")

# --- Load Processed Dataset ---
@st.cache_data
def load_data():
    try:
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
X = df[['price', 'num_lectures', 'content_duration', 'is_paid', 'level_encoded', 'subject_encoded']]
y = df['is_successful']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# --- Performance Metrics Section ---
with st.container():
    st.subheader("📈 Model Performance Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📍 Precision", f"{precision * 100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🚀 Recall", f"{recall * 100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💡 F1 Score", f"{f1 * 100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 AUC", f"{auc:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)


# --- Sidebar for Filters and Visualizations ---
st.sidebar.header("📁 Filters & Visualizations")
st.sidebar.markdown("### ⚙️ Dynamic Filters")
selected_segment = st.sidebar.selectbox("Filter by Customer Segment", ['All'] + list(df['subject'].unique()))
selected_tier = st.sidebar.selectbox("Filter by Service Tier", ['All'] + list(df['level'].unique()))

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
with st.container():
    st.subheader(f"📊 Visualization: {option}")
    if not filtered_df.empty:
        sns.set_theme(style="whitegrid")
        plt.style.use('seaborn-v0_8-whitegrid')

        if option == "Key Retention Drivers":
            st.write("Shows which customer features are most influential in predicting retention.")
            importances = model.feature_importances_
            feature_map = {'price': 'Service Cost', 'num_lectures': 'Interactions', 'content_duration': 'Service Usage', 'is_paid': 'On Paid Plan', 'level_encoded': 'Service Tier', 'subject_encoded': 'Customer Segment'}
            features_df = pd.DataFrame({'Feature': [feature_map.get(f, f) for f in X.columns], 'Importance': importances}).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=features_df, x='Importance', y='Feature', ax=ax, palette="viridis")
            ax.set_title("Key Retention Drivers", fontsize=16)
            st.pyplot(fig)

        elif option == "Segment vs Retention":
            st.write("Compares the number of retained vs. churned customers for each segment.")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=filtered_df, x='subject', hue='is_successful', ax=ax, palette='muted')
            plt.xticks(rotation=45, ha='right')
            ax.set_title("Customer Segment vs Retention Rate", fontsize=16)
            st.pyplot(fig)

        else: # Handle other graphs similarly
            # This is a placeholder for the rest of your graph logic
            st.write(f"Displaying graph for: {option}")

    else:
        st.warning("No data available for the selected filters.")


# --- High Retention Rate Customers ---
st.markdown("---")
with st.container():
    with st.expander("🏆 View Customers with High Predicted Retention Rate (>50%)", expanded=False):
        with st.spinner('Analyzing customers...'):
            all_customers_proba = model.predict_proba(X)[:, 1]
            df['retention_probability'] = all_customers_proba
            high_retention_customers = df[df['retention_probability'] > 0.5].sort_values(by='retention_probability', ascending=False)
            high_retention_customers['retention_probability_percent'] = high_retention_customers['retention_probability'] * 100
            st.dataframe(high_retention_customers[['course_title', 'subject', 'price', 'num_subscribers', 'retention_probability_percent']].rename(columns={
                'course_title': 'Customer ID', 'subject': 'Segment', 'price': 'Service Cost ($)',
                'num_subscribers': 'Usage Metric', 'retention_probability_percent': 'Retention Probability (%)'
            }))

# --- Live Prediction Form ---
st.markdown("---")
with st.container():
    st.subheader("🔮 Predict Customer Retention")
    st.write("Enter the details of a hypothetical customer to predict their retention.")
    with st.form("predict_form"):
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
with st.container():
    st.subheader("📝 Customer Feedback")
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
with st.container():
    st.subheader("📥 Download Data")
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

