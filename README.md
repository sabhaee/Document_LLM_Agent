# Document Agent : Chat with your Data
This repository contains an streamlit application for a document, question andswering agent, that leverag a pretrained large language models (LLMs) to provide in-context answer to user query's based on user data. Project usese a vector store and langchain pipeline to creat an agent that can chat with the uploaded documents.


## Getting Started

Follow these steps to run the Sentiment Analysis Demo:

1. Clone this repository to your local machine:
   ```shell
   git clone https://github.com/sabhaee/Document_LLM_Agent.git
   ```

2. Creat a virtual envoronment and install required dependencies 
    ```shell
    pip install -r requirements.txt
    ```

3. Change into the project directory:
4. Add you OpenAI API key in `.env` file
5. Start the application
    ```shell
    streamlit run app.py
    ```
    

## Usage

1. Upload single or multiple PDF documents.

2. Click the "Process" button to perform document processing and creating vector store.

3. The application will display a text input box for the user to enter the first query

4. Input your question and agent will provide an answer in naturaal language format from the uploaded documents 

5. A expandable text region shows the source and most relavant section of the document that was used to provide this answer

## Sample app output

![Screenshot](https://github.com/sabhaee/Document_LLM_Agent/blob/main/sample/Screenshot.png)


## Customization

- To use a different LLM model, you can edit the `intialize_llm` function in the `utility.py` file to use a differenct LLM model.

