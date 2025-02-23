import streamlit as st
import re
from crewai import Agent, Task, Crew, Process
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from crewai import LLM

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=220)
    split_docs = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    return vectorstore

class PDFRetrievalAgent(Agent):
    def __init__(self, retriever):
        super().__init__(
            role="PDF Retriever",
            backstory="I retrieve relevant information from research papers stored as PDFs.",
            goal="Fetch the most relevant document excerpts based on a user query."
        )
        self._retriever = retriever

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        query = task.description
        docs = self._retriever.invoke(query)
        retrieved_text = "\n".join([doc.page_content for doc in docs])
        return retrieved_text

class LLMAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="LLM Processor",
            backstory="I analyze and refine retrieved excerpts to generate meaningful insights.",
            goal="Summarize and interpret retrieved text to provide a clear response."
        )
        self.llm = llm
    
    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        if not context or not isinstance(context, str) or len(context.strip()) == 0:
            return "No relevant information found in the document."
        
        system_prompt = (
            "You are an AI assistant that answers ONLY based on the provided document excerpts. "
            "Do not use external knowledge. If the answer is not found, reply with 'Not found in the document.'\n\n"
            "DOCUMENT EXCERPTS:\n" + context
        )

        return self.llm.call([{"role": "system", "content": system_prompt}, {"role": "user", "content": task.description}])


def extract_think_content(text):
    match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""

# Function to remove everything between <think> and </think> tags
def remove_think_content(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# Streamlit UI
st.title("Agentic RAG: Document Query Assistant")
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    vectorstore = process_pdf("temp.pdf")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    st.sidebar.success("PDF uploaded and processed! ✅ Start asking questions below.")
    pdf_retrieval_agent = PDFRetrievalAgent(retriever)
    llm = LLM(model="ollama/deepseek-r1:1.5b", base_url="http://localhost:11434")
    llm_agent = LLMAgent(llm)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question about your document"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        retrieval_task = Task(description=prompt, expected_output="Relevant excerpts from the PDF.", agent=pdf_retrieval_agent)
        llm_task = Task(description="Analyze and summarize the retrieved text.", expected_output="A well-structured response.", agent=llm_agent)
        
        crew = Crew(agents=[pdf_retrieval_agent, llm_agent], tasks=[retrieval_task, llm_task], verbose=True, process=Process.sequential)
        result = crew.kickoff()

        answer = result.raw if result else "No relevant information found."

        result_content=remove_think_content(answer)
        show_think=extract_think_content(answer)
        st.session_state["messages"].append({"role": "assistant", "content": result_content})


        with st.chat_message("user"):
            st.write(prompt)

        if show_think:
            with st.expander("🤔 Show/Hide Reasoning"):
                st.write(show_think)
                

        with st.chat_message("assistant"):
            st.write(result_content)
