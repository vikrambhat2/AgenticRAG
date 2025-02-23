
# **Agentic RAG - Document Query System**

## **Overview**

This project implements a Retrieval-Augmented Generation (RAG) system that processes PDF documents to retrieve relevant information based on user queries. The system uses two agents:

1. **PDF Retrieval Agent**: Retrieves relevant document excerpts from a PDF based on a user query.
2. **LLM Agent**: Processes the retrieved excerpts to generate clear and structured responses.

The system is designed to load PDF documents, process them using embeddings, and then answer user queries by combining retrieval-based and generation-based techniques.

## **Features**

- **PDF Document Processing**: Extracts and processes text from PDF documents.
- **Document Chunking & Embedding**: Splits the document into chunks and generates embeddings for efficient retrieval.
- **Query Handling**: Users can ask questions, and the system will retrieve relevant document excerpts and generate a response.
- **Agent-Oriented Design**: Uses `crewai` agents to manage retrieval and processing tasks.
- **Streamlit UI**: A simple interface to interact with the system (if applicable).

## **Setup**

### **Dependencies**

To install the required dependencies, run:

```
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `streamlit`: For building the user interface.
- `crewai`: For creating and managing agents.
- `langchain`: For document processing, text splitting, and embeddings.
- `faiss-cpu`: For efficient document retrieval with FAISS.
- `huggingface_hub`: For using Hugging Face embeddings.

### **Required Models**

- **Embedding Model**: The project uses the `sentence-transformers/all-MiniLM-L6-v2` model for embeddings, which can be loaded from Hugging Face.
- **LLM Model**: The `ollama/deepseek-r1:1.5b` model is used for generating responses.

Ensure you have access to these models or replace them with appropriate alternatives.

## **Usage**

1. **Upload a PDF**: The system loads and processes the PDF file provided by the user.
2. **Ask a Question**: Users can input a query related to the document.
3. **Retrieval and Summarization**: 
   - The **PDF Retrieval Agent** retrieves relevant document excerpts.
   - The **LLM Agent** processes these excerpts to generate a structured response.
4. **View the Answer**: The final answer is displayed along with any reasoning extracted from the system.

### **Example Query**

To ask a question, simply input a query like:

```plaintext
"What are the prerequisites to run the project?"
```

The system will fetch relevant excerpts from the uploaded PDF and generate an insightful response.

## **How It Works**

1. **Process PDF**: The PDF is loaded and split into smaller chunks for easier retrieval.
2. **Vectorization**: The chunks are converted into embeddings and stored in a FAISS vector store.
3. **Retrieve Excerpts**: The **PDF Retrieval Agent** uses the vector store to find the most relevant excerpts based on the user's query.
4. **Generate Response**: The **LLM Agent** processes the retrieved text and generates a coherent response.

## **Contributions**

Feel free to fork the repository, contribute, or raise issues if you encounter any bugs or have suggestions for improvement.

