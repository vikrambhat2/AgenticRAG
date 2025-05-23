
# Smart Resume-JD Matcher with MCP and Ollama

This project demonstrates a lightweight multi-agent system powered by **MCP (Model Context Protocol)** and **Ollama** to analyze and compare resumes with job descriptions (JDs). It parses unstructured text into structured fields, matches relevant skills and experiences, and generates a professional gap analysis — all using local LLMs via Ollama.

> ✅ Ideal for showcasing MCP tooling workflows and LLM orchestration with minimal latency.

---

## 🚀 Features

- Resume and JD Parsing (Name, Skills, Experience, etc.)
- Skill Matching & Gap Identification
- Professional Summary Generation
- Powered by [FastMCP](https://pypi.org/project/fastmcp/) and [Ollama LLMs](https://ollama.com/)
- Streamlit-based front-end for interactive use

---

## 🧩 Tech Stack

- **FastMCP**: Lightweight server for defining LLM tools
- **LangChain + Ollama**: Local LLM invocation via `llama3.2`
- **Streamlit**: Interactive UI for uploading and analyzing documents
- **PyPDF2**: Extracts text from PDF resumes and JDs

---

## 📦 Installation

1. **Clone the repo**:

```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
```

2. **Install dependencies**:

   Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   Install Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run Ollama**:

   * Follow [Ollama installation guide](https://ollama.com/download).
   * Pull the model used in the project:

     ```bash
     ollama pull llama3
     ```

---

## 🛠️ Running the Project

### 1. Start the MCP Server

This script defines all four tools:

* `parse_resume`
* `parse_jd`
* `match_resume_to_jd`
* `summarize_gap`

Make sure:
- You have [Ollama](https://ollama.com) running locally.


#### Option A: Run in Development Mode (auto-reloads on changes)
```bash
mcp dev mcp_server.py
```bash
python mcp_server.py
```
####  Option B: Run in Production Mode
```bash
mcp run mcp_server.py
```

> You should see logs indicating that the MCP server is running on `http://127.0.0.1:6274`.

---

### 2. Launch the Streamlit UI

Update `MCP_URL` in `mcp_client_streamlitUI.py` to point to your MCP server if required (usually `http://127.0.0.1:6274`):

```python
MCP_URL = "http://127.0.0.1:6274"
```

Then run:

```bash
streamlit run mcp_client_streamlitUI.py
```

---

## 🧪 Usage

1. Upload or paste resume and JD text in the Streamlit interface.
2. Click "Match Resume to JD".
3. View:

   * Parsed fields
   * Skill match/mismatch
   * Business-style gap summary

---

## 📁 Project Structure

```
.
├── mcp_server.py                # Defines tools and launches MCP server
├── mcp_client_streamlitUI.py   # Streamlit frontend that talks to MCP tools
├── requirements.txt
└── README.md
```

---

## 📌 Notes

* **MCP** is a lightweight protocol layer for orchestrating tools, chainable prompts, and LLM outputs.
* You can replace `llama3.2` with any local model in Ollama that supports chat format.
* If Ollama is not running or the model isn’t pulled, the MCP server will fail on startup.

---

## 🧠 Future Enhancements

* Integrate feedback loops (e.g., refine skills)
* Multi-turn interactions (with memory)
* Candidate ranking against multiple JDs

---

