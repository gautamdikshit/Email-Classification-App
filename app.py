import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load the pre-trained model and tokenizer
save_directory = "/content/drive/MyDrive/SpamHamModel"
model = TFBertForSequenceClassification.from_pretrained(save_directory)
tokenizer = BertTokenizer.from_pretrained(save_directory)

# Streamlit app
st.title("Spam/Ham Email Classifier")

# Input text box for user to enter email content
email_input = st.text_area("Enter the email text:")

# Predict button
if st.button("Classify Email"):
    if email_input:
        # Tokenize the input email text
        input_encodings = tokenizer([email_input], truncation=True, padding=True, return_tensors="tf")
        
        # Make predictions using the loaded model
        predictions = model(input_encodings)
        
        # Get the predicted class (Spam: 1, Ham: 0)
        predicted_class = tf.argmax(predictions.logits, axis=1).numpy()[0]
        
        # Show the result
        if predicted_class == 1:
            st.write("**Prediction**: Spam")
        else:
            st.write("**Prediction**: Ham")
    else:
        st.write("Please enter an email text to classify.")
