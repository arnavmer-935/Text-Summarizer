import streamlit as st
from main import read_pdf_text
from main import text_summary
import os
from main import divide_text_chunks


st.set_page_config("Text Summarizer")
st.header("Chat with Google Gemini Pro LLM Model and get your Summary")

user_input = st.text_area("Please Provide the Complete text...")
language_options = ['English', 'Hindi', 'Marathi', 'Spanish']
selected_language = st.selectbox("Select Language:", language_options, index=language_options.index('English'))
button = st.button("Submit And Get Summary..")
if user_input and button:
    response = text_summary(user_input,selected_language)
    st.write(response)

response =""
with st.sidebar:
    st.title("Uploaded a Pdf File...")
    pdf_docs = st.file_uploader("Upload your files in .pdf format and Click on Submit and process your files")
    button = st.button("Submit and Process")
    if(button):
         raw_text = read_pdf_text(pdf_docs)
         response = divide_text_chunks(raw_text)
st.write(response)
         

