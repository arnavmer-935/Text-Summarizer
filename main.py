import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
import os


load_dotenv()
genai.configure(api_key = GOOGLE_API_KEY)
llm_model = ChatGoogleGenerativeAI(model = "gemini-pro",temperature=0.4)

def read_pdf_text(pdf_docs):   
    # provide the path of  pdf file/files.
    pdfreader = PdfReader(pdf_docs)
    # read text from pdf
    text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            text += content
    return text


def divide_text_chunks(text):
    ## Splittting the text
    chunks_prompt="""
    Please summarize the below text:
    text:`{text}'
    Summary:
    """
    map_prompt_template=PromptTemplate(input_variables=['text'],
                                        template=chunks_prompt)


    final_combine_prompt='''
    Provide a final summary of the entire text with these important points.
    Add a Generic Motivational Title,
    Start the precise summary with an introduction and provide the
    summary in few paragraps for the text based on the length of the given text.
    include the observation,results,conclusions sections also.
    try to include any important equations or mathematical formulas if you find in the text.
    Speech: `{text}`
    '''
    final_combine_prompt_template=PromptTemplate(input_variables=['text'],
                                                template=final_combine_prompt)



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    chunks = text_splitter.create_documents([text])


    global llm_model
    chain = load_summarize_chain(llm_model , chain_type='map_reduce' , map_prompt=map_prompt_template,
    combine_prompt=final_combine_prompt_template,verbose=False)



    summary = chain.run(chunks)
    return summary


def text_summary(text,language="english"):
    generic_template='''
    Write a detailed summary of the following text:
    Context : {text}
    Translate the precise summary to {language}.
    '''

    prompt=PromptTemplate(
        input_variables=['text','language'],
        template=generic_template
    )

    global llm_model
    llm_chain=LLMChain(llm=llm_model,prompt=prompt)

    summary=llm_chain.invoke({'text':text,'language':language})
    return summary
  

