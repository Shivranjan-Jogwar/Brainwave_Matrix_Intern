import streamlit as st
import joblib

# Load the model and vectorizer (make sure these files are in the same directory)
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("Twitter Sentiment Analysis 🚀")
st.write("Enter a tweet to predict its sentiment:")

# Input box
tweet = st.text_input("Tweet goes here:")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        # Vectorize the input
        X_input = vectorizer.transform([tweet])

        # Predict
        prediction = model.predict(X_input)[0]

        # Output
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
        st.success(f"Predicted Sentiment: {sentiment}")