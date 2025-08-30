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
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

llm_model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.4)

def read_pdf_text(pdf_docs):   
    pdfreader = PdfReader(pdf_docs)
    text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            text += content
    return text


def divide_text_chunks(text):
    chunks_prompt="""
    Please summarize the below text:
    text:`{text}'
    Summary:
    """
    map_prompt_template=PromptTemplate(input_variables = ['text'],
                                        template = chunks_prompt)

    final_combine_prompt = '''
    Summarize the input text with the following goals:
    Retain key points, main ideas, and essential information.
    Eliminate redundancy, filler words, and minor details.
    Maintain clarity and logical flow.
    Format: Bullet pointers with clearly segregated sub-headings for each major topic within the text or document.
    Tone: Neutral, concise, and professional.
    Detail: Include the following, if applicable to the subject of the user's input:
        English & Grammar	- Rules and examples
                            - Key concepts (e.g., sentence structure)
                            - Common mistakes and how to avoid them
                            
        Literature	- Themes and messages
                    - Plot summary
                    - Key characters and roles
                    - Literary devices (e.g., metaphors)
                    - Historical context
                    
        Science	- Key concepts and theories
                - Diagrams (e.g., labeled charts)
                - Equations and formulas
                - Applications
                - Experiments
                
        Economics	- Definitions and key terms
                    - Graphs and charts (e.g., demand-supply curve)
                    - Data and figures
                    - Economic theories
                    - Examples
                    
        Engineering	- Principles and frameworks
                    - Diagrams (e.g., circuit designs)
                    - Equations
                    - Applications
                    - Innovations and advancements
                    
        History	- Timelines
                - Key events and turning points
                - Influential people
                - Causes and effects
                - Facts and dates
                
        Social Science	- Key theories (e.g., sociology concepts)
                        - Case studies
                        - Figures and trends
                        - Important terminology
                        
        Mathematics	- Equations and formulas
                    - Step-by-step methods
                    - Key concepts 
                    - Diagrams and graphs
                    - Applications
                    
        Programming	- Code snippets
                    - Algorithms and workflows
                    - Key concepts (e.g., OOP, data structures)
                    - Use cases
                    
        Data Science - Definitions and techniques (e.g., regression, clustering)
                     - Equations
                     - Examples (real-world datasets)
                     - Visual aids
                     
        Current Affairs	- Event summary (what, when, where, why)
                        - Key players involved
                        - Impact (short- and long-term)
                        - Facts and figures
                        - Analysis
                        
        Law	- Legal terms and definitions
            - Landmark cases and judgments
            - Key principles and sections of law
            - Structure (articles, clauses)
            - Real-life applications
            
        Logical Reasoning	- Concepts (e.g., deductive reasoning)
                            - Examples of reasoning problems
                            - Steps to solve
                            - Visual aids (flowcharts, diagrams)
                            
    Adaptability: Adjust the summary length and detail based on the input text's complexity and purpose (e.g., article, research paper, or business report).
    Speech: `{text}`
    '''
    final_combine_prompt_template = PromptTemplate(input_variables = ['text'],
                                                template = final_combine_prompt)



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    chunks = text_splitter.create_documents([text])


    global llm_model
    chain = load_summarize_chain(llm_model, chain_type = 'map_reduce', map_prompt = map_prompt_template,
    combine_prompt = final_combine_prompt_template, verbose = False)

    summary = chain.run(chunks)
    return summary


def text_summary(text, language="english"):
    generic_template = '''
    Write a detailed summary of the following text:
    Context : {text}
    Translate the precise summary to {language}.
    '''

    prompt = PromptTemplate(
        input_variables = ['text','language'],
        template = generic_template
    )

    global llm_model
    llm_chain = LLMChain(llm = llm_model, prompt = prompt)

    summary = llm_chain.invoke({'text':text,'language':language})
    return summary["text"]
  


