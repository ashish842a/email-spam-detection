import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the trained model
model = joblib.load("spam.pkl")

# Load the CountVectorizer
vectorizer = joblib.load("vectorizer.pkl")

def predict_spam(email_text):
    # Transform the input text into a feature vector
    X = vectorizer.transform([email_text])
    # Predict whether the email is spam or not
    prediction = model.predict(X)
    return prediction[0]

# Streamlit app
def main():
    st.title("Spam Email Detection")

    # Input field for user to enter email text
    email_text = st.text_area("Enter email text:", "")

    if st.button("Detect"):
        if email_text.strip() == "":
            st.error("Please enter some text.")
        else:
            # Predict whether the email is spam or not
            prediction = predict_spam(email_text)
            if prediction == 1:
                st.error("This email is classified as spam.")
            else:
                st.success("This email is classified as not spam.")

if __name__ == "__main__":
    main()
