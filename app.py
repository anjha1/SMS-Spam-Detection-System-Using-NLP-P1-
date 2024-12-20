import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('spam_classifier_model.pkl', 'rb') as f:
    spam_classifier_model = pickle.load(f)

# Streamlit app
st.title('SMS Spam Detection App')

# Input text box
user_input = st.text_area('Enter the SMS text to classify:', '')

if st.button('Classify'):
    if user_input.strip():
        # Transform the input text using the vectorizer
        input_tfidf = tfidf_vectorizer.transform([user_input])

        # Predict using the model
        prediction = spam_classifier_model.predict(input_tfidf)[0]

        # Display the result
        if prediction == 1:
            st.error('This message is classified as SPAM.')
        else:
            st.success('This message is classified as HAM (not spam).')
    else:
        st.warning('Please enter a valid SMS text.')
