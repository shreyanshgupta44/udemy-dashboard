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
st.set_page_config(page_title="Udemy Course Dashboard", layout="wide")

# --- Title ---
st.title("📚 Udemy Course Success Predictor & Visualizer")

# --- Load Processed Dataset ---
# We now load the new dataset which already has the necessary columns.
try:
    df = pd.read_csv("processed_udemy_data.csv")
except FileNotFoundError:
    st.error("Error: 'processed_udemy_data.csv' not found. Please make sure the file is in the same directory.")
    st.stop() # Stop the app if the file is not found

# --- Re-initialize LabelEncoders for Prediction Form ---
# We still need to create LabelEncoder objects to transform the user's live input later.
# We will fit them on the unique values from the dataframe columns.
level_le = LabelEncoder()
df['level_encoded'] = level_le.fit_transform(df['level'])

subject_le = LabelEncoder()
df['subject_encoded'] = subject_le.fit_transform(df['subject'])


# --- Model Training ---
# Define features (X) and target (y)
X = df[['price', 'num_lectures', 'content_duration', 'is_paid', 'level_encoded', 'subject_encoded']]
y = df['is_successful']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(random_state=42) # Added random_state for reproducibility
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# --- Performance Metrics ---
# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# Display metrics in columns
st.markdown("### 📈 Model Performance Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("📍 Precision", f"{precision * 100:.2f}%")
col3.metric("🚀 Recall", f"{recall * 100:.2f}%")
col4.metric("💡 F1 Score", f"{f1 * 100:.2f}%")
col5.metric("📊 AUC", f"{auc:.2f}")


# --- Sidebar for Visualizations ---
st.sidebar.title("📁 Visualizations")
option = st.sidebar.selectbox("Select a graph", [
    "Feature Importance", "Subject vs Success", "Level Distribution", "Price Distribution"
])


# --- Main Panel for Visualizations ---
st.markdown("---") # Visual separator
st.subheader(f"📊 Visualization: {option}")

if option == "Feature Importance":
    st.write("Shows which course features are most influential in predicting success.")
    importances = model.feature_importances_
    features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=features_df, x='Importance', y='Feature', ax=ax, palette="viridis")
    ax.set_title("Feature Importance from Random Forest", fontsize=16)
    st.pyplot(fig)

elif option == "Subject vs Success":
    st.write("Compares the number of successful vs. unsuccessful courses for each subject.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='subject', hue='is_successful', ax=ax, palette='pastel')
    plt.xticks(rotation=45, ha='right')
    ax.set_title("Subject vs Success Rate", fontsize=16)
    st.pyplot(fig)

elif option == "Level Distribution":
    st.write("Shows the distribution of courses across different difficulty levels.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='level', ax=ax, palette="plasma")
    ax.set_title("Course Level Distribution", fontsize=16)
    st.pyplot(fig)

elif option == "Price Distribution":
    st.write("Displays the frequency of different course prices (for courses under 200).")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df[df['price'] < 200], x='price', bins=30, kde=True, ax=ax, color='purple')
    ax.set_title("Price Distribution (Under 200)", fontsize=16)
    st.pyplot(fig)


# --- Live Prediction Form ---
st.markdown("---")
st.markdown("## 🔮 Try Live Prediction")

with st.form("predict_form"):
    st.write("Enter the details of a hypothetical course to predict its success.")
    
    # Create columns for a cleaner layout
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        price = st.number_input("Course Price ($)", min_value=0, max_value=500, value=100)
        num_lectures = st.number_input("Number of Lectures", min_value=1, max_value=1000, value=30)
        duration = st.number_input("Content Duration (hours)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

    with form_col2:
        is_paid_str = st.selectbox("Is the course Paid?", ["Yes", "No"])
        level = st.selectbox("Course Level", options=level_le.classes_)
        subject = st.selectbox("Subject", options=subject_le.classes_)

    submit = st.form_submit_button("Predict Success")

    if submit:
        # Transform user input using the fitted encoders
        level_encoded_input = level_le.transform([level])[0]
        subject_encoded_input = subject_le.transform([subject])[0]
        is_paid_input = 1 if is_paid_str == "Yes" else 0

        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'price': [price],
            'num_lectures': [num_lectures],
            'content_duration': [duration],
            'is_paid': [is_paid_input],
            'level_encoded': [level_encoded_input],
            'subject_encoded': [subject_encoded_input]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        # Display result
        if prediction == 1:
            st.success(f"✅ This course is likely to be **successful**! (Probability: {prediction_proba:.2%})")
        else:
            st.warning(f"⚠️ This course may **not perform well**. (Success Probability: {prediction_proba:.2%})")


# --- Download Processed Data ---
st.markdown("---")
st.markdown("### 📥 Download Processed Data")

# Cache the conversion to prevent re-running on every interaction
@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df)

st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name='processed_udemy_data.csv',
    mime='text/csv',
)
