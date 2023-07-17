import tempfile
import time
import os
import streamlit as st
from dotenv import load_dotenv
import utility



# Set the title and subtitle of the app
st.title('💬🔎🔗 Document Agent: Ask Questions From Your Documents')
st.subheader('Load PDF documents, retrieve information by asking questions!')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'processed' not in st.session_state:
    st.session_state.processed = []



# Loading the PDF document files
st.subheader('Upload Your Documents')
uploaded_files = st.file_uploader('file_uploader', type=(['pdf']), accept_multiple_files=True,label_visibility="collapsed")


temp_file_path = os.getcwd()

# Load environment variables from .env
load_dotenv()
# Set APIkey for OpenAI Service
openai_api_key = os.getenv("OPENAI_API_KEY")
# Intialize embedding model
embeddings = utility.embedding_model(openai_api_key)


# Button to initiate file processing
if st.button('Process Files') and uploaded_files:
    
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        st.write("Full path of the uploaded file:", temp_file_path)
    with st.spinner('Processing...'):
        docsearch = utility.process_documents(temp_dir.name,embeddings)
        
    st.session_state.processed = docsearch
    
# Intialize LLM
llm = utility.intialize_llm(openai_api_key)

if st.session_state.processed:
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 
    
    agent_executor = utility.creat_agent(st.session_state.processed, llm)
    
    if prompt := st.chat_input("Input your question here"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        

        # Querry agent with prompt
        llm_response = agent_executor.run(prompt)
        
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate stream of response with milliseconds delay
            for chunk in llm_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        
            with st.expander('Most Relevant Source in Documents'):
                # # Find the relevant pages
                search = st.session_state.processed.similarity_search_with_score(prompt) 
                
                st.write(search[0][0].page_content) 

                file_name = search[0][0].metadata['source'].split("\\")
                st.write(f"Source: {file_name[-1]}")
    
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        if len(st.session_state.messages)>0 and st.button('Clear Chat history'):
                    st.session_state.messages = []
        