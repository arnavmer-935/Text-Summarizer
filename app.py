import streamlit as st
from main import read_pdf_text
from main import text_summary
import os
from main import divide_text_chunks


st.set_page_config("Text Summarizer")
st.header("TEXT SUMMARIZER")
st.header("Upload any file/URL or input any text to generate a summary in a language of your choice")

user_input = st.text_area("Please enter input text here")
language_options = ["English", "Mandarin", "Spanish", "Hindi", "Bengali", "Portuguese", "Russian", 
    "Japanese", "German", "French", "Turkish", "Italian", "Korean", "Vietnamese", 
    "Tamil", "Urdu", "Persian", "Arabic", "Dutch", "Polish", "Thai", "Swedish", 
    "Czech", "Greek", "Hungarian", "Romanian", "Bulgarian", "Hebrew", "Danish", 
    "Norwegian", "Finnish", "Malay", "Indonesian", "Tagalog", "Serbian", "Croatian", 
    "Slovak", "Slovenian", "Lithuanian", "Latvian", "Estonian", "Georgian", "Armenian", 
    "Azerbaijani", "Kazakh", "Uzbek", "Turkmen", "Mongolian", "Pashto", "Sindhi", 
    "Punjabi", "Gujarati", "Marathi", "Kannada", "Malayalam", "Telugu", "Sinhala", 
    "Nepali", "Burmese", "Khmer", "Lao", "Bislama", "Fijian", "Tongan", "Samoan", 
    "Maori", "Hawaiian", "Yoruba", "Hausa", "Swahili", "Zulu", "Xhosa", "Amharic", 
    "Tigrinya", "Somali", "Wolof", "Fula", "Igbo", "Shona", "Sesotho", "Tswana", 
    "Herero", "Afrikaans", "Chewa", "Mandinka", "Javanese", "Sundanese", "Balinese", 
    "Madurese", "Minangkabau", "Acehnese", "Tetum", "Papiamento", "Quechua", "Aymara", 
    "Guarani", "Mapudungun", "Nahuatl"]
    
selected_language = st.selectbox("Select Language:", language_options, index = language_options.index('English'))
button = st.button("Submit And Get Summary")
if user_input and button:
    response = text_summary(user_input, selected_language)
    st.write(response)

response = ""
with st.sidebar:
    st.title("Uploaded any PDF of upto 200MB:")
    pdf_docs = st.file_uploader("Upload your files in .pdf format and Click on \"Submit and Process\" to process your files.")
    button = st.button("Submit and Process")
    if(button):
         raw_text = read_pdf_text(pdf_docs)
         response = divide_text_chunks(raw_text)
st.write(response)
         

