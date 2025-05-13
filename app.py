import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---- Load and prepare data ----

# Load your fake and true datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "TRUE"

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Keep only necessary columns (update column name if needed)
df = df[['text', 'label']].dropna()

# Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ---- Streamlit App Interface ----

st.title("📰 Fake News Detector")
st.write("Enter a news article below and find out if it's **True** or **Fake**!")

# Text input
user_input = st.text_area("Enter News Text Here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        st.subheader("🧠 Prediction Result:")
        if prediction == "FAKE":
            st.error("🚨 This news is likely **FAKE**.")
        else:
            st.success("✅ This news is likely **TRUE**.")

# Optional: display some sample data
with st.expander("📊 Show some sample data"):
    st.write(df.sample(5))
