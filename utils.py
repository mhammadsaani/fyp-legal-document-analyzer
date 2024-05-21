import os
import time 

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

from api_key import token_key_hf


os.environ["HUGGINGFACEHUB_API_TOKEN"] = token_key_hf
HUGGINGFACEHUB_API_TOKEN = token_key_hf

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}


def prepare_models():
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_new_tokens=1000, temperature=0.01, token=HUGGINGFACEHUB_API_TOKEN
    )
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return llm, embeddings

llm, embeddings = prepare_models()


def response_generator(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)


def extract_relevant_prompts(file_name):
    with open(f'prompts/{file_name}.txt', 'r') as file:
        content = file.read()
    prompts = content.split('-----')
    return prompts[0], prompts[1]


def get_mistral_instruction(document_text):
    mistral_instruction = f"""
    ### Instruction:
    '
    Please provide a summary of the Non-Disclosure Agreement (NDA) document in plain, easy-to-understand language, paying close attention to the Clauses present in the NDA. A clause is a specific point or provision in a law or legal document like a non-disclosure agreement. It can be an article, section, or standalone paragraph that addresses any topic about the document that contains it.  The summary of the document should be clause-wise. The format of the summary should be the following, 
    "
    General Information About the Document: Paragraph with General Information about the Document.
    1. Clause Name: 
    ----------------- Summary of the Clause in Very Easy and plain language. The summary should assume that the user doesn't have any information about legal terminology and should explain the terminology if needed.  The summary should be comprehensive and should not miss the necessary details.
    2. Clause Name: 
    ----------------- Summary of the Clause in Very Easy and plain language. The summary should assume that the user doesn't have any information about legal terminology and should explain the terminology if needed.  The summary should be comprehensive and should not miss the necessary details.
    and the rest of the clauses.
    Once all clauses are summarized, the last paragraph should be Implications if you signed the document: This paragraph should include that after signing this document you are bound to the following things. The language of this paragraph should be the simplest. This should be a detailed paragraph taking information from all the clauses mentioned above.
    "
    Don't use any legal jargon, not even a single complex word should be used. You should take this thing into account. Everything should be in the simplest form possible.
    '

    ### Input:
    {document_text}

    ### Response:
    """
    return mistral_instruction







def fetch_result(document_text, feature):
    if feature == 'extract_clauses':
        sys_message, human_message = extract_relevant_prompts('clause_extraction_prompt')
    elif feature == 'generate_summary':
        sys_message, human_message = extract_relevant_prompts('generate_summary_prompt')
    elif feature == 'extract_entities':
        sys_message, human_message = extract_relevant_prompts('entities_extraction_prompt')
        
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', sys_message),
            ('human', human_message)
        ]
    )
    
    chain = chat_prompt | llm  | StrOutputParser()
    return chain.invoke({"document_text": document_text})
