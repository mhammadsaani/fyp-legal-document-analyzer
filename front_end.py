import os
import pickle

from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma

from utils import embeddings, llm, fetch_result, response_generator


system_message = """Please answer the user question by keeping in view that You are an expert assistant specializing in Non-Disclosure Agreements (NDAs). Your role is to answer queries strictly related to NDAs. 
Provide precise and accurate information about what is being asked related to NDA. Provide the answer in very plain language even a person knowing
nothing regarding Legal Terms should be able to understand. If the answer is nto related, then strictly  answer with the following term 'Ask Question from The Document'"""


with st.sidebar:
    st.title('Legalysis')
    st.markdown('''
    ## About
    An LLM-powered Legal Assistant using
        - Mistral
        - Langchain
        - Streamlit

    ''') 
    summary = st.button('Summarize') 
    clauses = st.button('Extract Clauses')
    entities = st.button('Extract Entities') 
    add_vertical_space(4)
    

def features_response(prompt, history, text, feature):
    st.chat_message("user").markdown(prompt)
    history.append({"role": "user", "content": prompt})
    if feature == 'summary':
        response = f"{fetch_result(text, 'generate_summary')}"
    elif feature == 'clauses':
        response = f"{fetch_result(text, 'extract_clauses')}"
    elif feature == 'entities':
        response = f"{fetch_result(text, 'extract_entities')}"
    history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        response = response_generator(response)
        st.write_stream(response)


def main():
    st.header('Do conversation with legal bot')
    pdf = st.file_uploader('Upload a Legal Document', type='pdf')
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=1000,
            length_function=len, 
        )
        
        chunks = text_splitter.split_text(text=text)
        file_name = pdf.name[:-4]
        
        if os.path.exists(f'pickle_files/{file_name}.pkl'):
            with open(f'pickle_files/{file_name}.pkl', 'rb') as file:
                VectorStore = pickle.load(file)
        else:
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f'pickle_files/{file_name}.pkl', 'wb') as file:
                pickle.dump(VectorStore, file)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
     
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
             
        if summary:  
            prompt = 'Summarize the document'
            features_response(prompt, st.session_state.messages, text, feature='summary')
                
        if clauses:
            prompt = 'Extract the Clauses from the document'
            features_response(prompt, st.session_state.messages, text, feature='clauses')
            
        if entities:
            prompt = "Extract entities from the document"
            features_response(prompt, st.session_state.messages, text, feature='entities')

 
        prompt = st.chat_input("Any Questions related to Legal Document?")
        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            docs = VectorStore.similarity_search(query=prompt)            
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            prompt = f"{system_message}. Here is the question {prompt}"
            response = chain.run(input_documents=docs, question=prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                response = response_generator(response)
                st.write_stream(response)
        
        
if __name__ == '__main__':
    main()
    