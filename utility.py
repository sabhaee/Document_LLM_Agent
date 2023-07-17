import os
from langchain.agents.agent_toolkits import create_vectorstore_agent,VectorStoreToolkit,VectorStoreInfo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

def process_documents(document_folder_PATH,embeddings):
    """
    This function load, split the documnets into chunks then creat 
    embedding of the text pieces and creat a vector store

    Args:
        document_folder_PATH: Path to folder where documents are saved
        embeddings: model to create text embeddings
    Return:
        Vector Store of embeding indexes


    """
    # Load documents
    loader = DirectoryLoader(document_folder_PATH, glob='**/*.pdf')
    # Load up your text into documents
    documents = loader.load()
    # Get your text splitter ready
    document_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split your documents into texts
    text_chunks = document_splitter.split_documents(documents)
    print(len(text_chunks))
    # Get your docsearch ready
    vector_index_db = FAISS.from_documents(text_chunks, embeddings)
    return vector_index_db

def save_vector_db(vector_db,db_name):
    vector_db.save_local(db_name)

def load_local_vector_db(db_name,embedding_model):
    FAISS.load_local(db_name, embedding_model)

def intialize_llm(api_key=None):
    # Initialize the LLM
    llm = OpenAI(temperature=0.1, verbose=True,openai_api_key=api_key)
    return llm

def embedding_model(api_key=None):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings

def creat_agent(vector_db,llm):
    # Create vectorstore info object
    vectorstore_info = VectorStoreInfo(
        name="Document vector",
        description=" Reference documents to answer user's queries",
        vectorstore= vector_db
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    return agent_executor
