import streamlit as st
import json
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM, Process

# Load environment variables
load_dotenv()

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# ------------------- Agents -------------------
from ResumeEnhancerAgent import ResumeEnhancerAgent
from ResumeParsingAgent import ResumeParsingAgent
from JDUnderstandingAgent import JDUnderstandingAgent
from MatchingAgent import MatchingAgent
from CoverLetterAgent import CoverLetterAgent
# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="🧠 Resume-JD Matcher", layout="wide")
st.markdown("# 🧠 AI Resume vs Job Description Matcher")
st.markdown("Use AI agents to evaluate how well a candidate’s resume matches a job description.")

with st.expander("ℹ️ Instructions", expanded=False):
    st.markdown("""
        1. Upload or paste both a **Resume** and a **Job Description**.
        2. Click one of the action buttons to:
           - **Run Matching**: Analyze resume and job description for a match score.
           - **Enhance Resume**: Get suggestions to improve the resume.
           - **Generate Cover Letter**: Create a tailored cover letter.
        3. Results will appear below the buttons.
    """)

st.markdown("---")
# Dependency Graph

with st.expander("### 🔄 Agent Dependency Graph", expanded=False):
    if GRAPHVIZ_AVAILABLE:  # Replace with GRAPHVIZ_AVAILABLE check
        dot = graphviz.Digraph(comment="Resume-JD Matcher Agent Workflow")
        dot.attr(rankdir='LR', size='10,8')

        # === User Inputs ===
        dot.node('U1', '👤 Resume Upload', shape='cylinder', color='orange', style='filled')
        dot.node('U2', '👤 JD Upload', shape='cylinder', color='orange', style='filled')

        # === Orchestrator ===
        dot.node('O', '🧭 Orchestrator', shape='box', style='filled', color='lightblue')

        # === Parsers ===
        dot.node('R', '📄 Resume Parser', shape='component')
        dot.node('J', '📑 JD Parser', shape='component')

        # === Core Agents ===
        dot.node('M', '🤝 Matcher', shape='box')
        dot.node('E', '🛠️ Enhancer', shape='box')
        dot.node('C', '✉️ Cover Letter', shape='box')
        dot.node('X', '🧠 Recommendation', shape='box')

        # === Final Outputs (clustered) ===
        with dot.subgraph(name='cluster_outputs') as c:
            c.attr(style='dashed', color='green', label='📦 Final Deliverables')
            c.node_attr.update(style='filled', color='lightgreen')
            c.node('MR', '📋 Match Report')
            c.node('RE', '📄 Enhanced Resume')
            c.node('CL', '✉️ Cover Letter')
            c.node('RC', '🧠 Recommendation Summary')

        # === Edges: Flow & Labels ===
        # User → Orchestrator
        dot.edge('U1', 'O', label='Resume')
        dot.edge('U2', 'O', label='Job Description')

        # Orchestrator → Parsers
        dot.edge('O', 'R', label='Route Resume')
        dot.edge('O', 'J', label='Route JD')

        # Parsers → Matcher
        dot.edge('R', 'M', label='Parsed Resume')
        dot.edge('J', 'M', label='Parsed JD')

        # Matcher → Match Report
        dot.edge('M', 'MR', label='Match Report')

        # Parsers → Enhancer
        dot.edge('R', 'E', label='Resume')
        dot.edge('J', 'E', label='JD')

        # Enhancer → Enhanced Resume
        dot.edge('E', 'RE', label='Improved Resume')

        # Parsers → Cover Letter
        dot.edge('R', 'C', label='Resume')
        dot.edge('J', 'C', label='JD')

        # Cover Letter → Output
        dot.edge('C', 'CL', label='Cover Letter')

        # Matcher → Recommendation Agent
        dot.edge('M', 'X', label='Match Insights')

        # Recommendation → Output
        dot.edge('X', 'RC', label='Recommendation')

        # Render graph
        st.graphviz_chart(dot)

    else:
        st.warning("Graphviz not installed. Please install it with `pip install graphviz` and ensure Graphviz software is installed. Showing placeholder instead.")
        st.image("https://via.placeholder.com/600x200?text=Agent+Dependency+Graph", caption="Agent Workflow (Install graphviz for full visualization)")

# --- Input Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Resume Input")
    resume_method = st.radio("Choose input method:", ["Upload PDF", "Paste Text"], key="resume_input")
    resume_text = ""
    if resume_method == "Upload PDF":
        uploaded_resume = st.file_uploader("Upload Resume", type=["pdf"], key="resume_pdf")
        if uploaded_resume:
            resume_text = extract_text_from_pdf(uploaded_resume)
    else:
        resume_text = st.text_area("Paste Resume Text", height=250)

    if resume_text:
        with st.expander("🔍 Resume Preview"):
            st.text_area("Extracted Resume Text", resume_text, height=150)

with col2:
    st.subheader("📑 Job Description Input")
    jd_method = st.radio("Choose input method:", ["Upload PDF", "Paste Text"], key="jd_input")
    jd_text = ""
    if jd_method == "Upload PDF":
        uploaded_jd = st.file_uploader("Upload JD", type=["pdf"], key="jd_pdf")
        if uploaded_jd:
            jd_text = extract_text_from_pdf(uploaded_jd)
    else:
        jd_text = st.text_area("Paste JD Text", height=250)

    if jd_text:
        with st.expander("🔍 JD Preview"):
            st.text_area("Extracted JD Text", jd_text, height=150)

st.markdown("---")

# --- Action Buttons and Output ---
if resume_text and jd_text:
    # Initialize LLM and agents
    try:
        llama_model = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        st.stop()

    resume_parser = ResumeParsingAgent(llm=llama_model)
    jd_parser = JDUnderstandingAgent(llm=llama_model)
    matcher = MatchingAgent(llm=llama_model)
    resume_enhancer = ResumeEnhancerAgent(llm=llama_model)
    cover_letter_generator = CoverLetterAgent(llm=llama_model)

    # Define tasks
    resume_task = Task(
        description=resume_text,
        expected_output="Structured resume data",
        agent=resume_parser
    )

    jd_task = Task(
        description=jd_text,
        expected_output="Structured JD data",
        agent=jd_parser
    )

    matching_task = Task(
        description="Match resume to JD.",
        expected_output="Match score and insights.",
        agent=matcher,
        context=[resume_task, jd_task]
    )

    resume_enhancer_task = Task(
        description="Optimize resumes based on job descriptions.",
        expected_output="Readable text report with improvements to the resume to make it more relevant to the JD.",
        agent=resume_enhancer,
        context=[resume_task, jd_task]
    )

    cover_letter_task = Task(
        description="Generate a cover letter based on the resume and job description.",
        expected_output="Readable text cover letter tailored to the job description.",
        agent=cover_letter_generator,
        context=[resume_task, jd_task]
    )

    # Create a placeholder for the output
    output_placeholder = st.empty()

    # Create columns for action buttons
    col3, col4, col5 = st.columns(3)

    with col3:
        if st.button("🚀 Run Matching"):
            with output_placeholder.container():
                with st.spinner("Analyzing resume and job description..."):
                    try:
                        crew = Crew(
                            agents=[resume_parser, jd_parser, matcher],
                            tasks=[resume_task, jd_task, matching_task],
                            name="Resume-JD Matcher Crew",
                            description="A team to analyze resumes and match them to job descriptions.",
                            verbose=True,
                            process=Process.sequential
                        )
                        result = crew.kickoff()
                        st.success("✅ Matching Complete")
                        try:
                            parsed_result = json.loads(result.raw)
                            st.markdown("### 📊 Match Summary")
                            st.markdown(parsed_result["match_summary"])
                        except Exception:
                            st.markdown("### 📊 Match Summary")
                            st.markdown(result.raw)
                    except Exception as e:
                        st.error(f"⚠️ Matching process failed: {str(e)}")

    with col4:
        if st.button("📝 Enhance Resume"):
            with output_placeholder.container():
                with st.spinner("Generating resume enhancements..."):
                    try:
                        crew = Crew(
                            agents=[resume_parser, jd_parser, resume_enhancer],
                            tasks=[resume_task, jd_task, resume_enhancer_task],
                            name="Resume Enhancer Crew",
                            description="A team to suggest resume improvements.",
                            verbose=True,
                            process=Process.sequential
                        )
                        result = crew.kickoff()
                        st.success("✅ Resume Enhancement Complete")
                        try:
                            parsed_result = json.loads(result.raw)
                            st.markdown("### 📝 Resume Enhancement Suggestions")
                            st.markdown(parsed_result["resume_enhancement"])
                        except Exception:
                            st.markdown("### 📝 Resume Enhancement Suggestions")
                            st.markdown(result.raw)
                    except Exception as e:
                        st.error(f"⚠️ Resume enhancement failed: {str(e)}")

    with col5:
        if st.button("✉️ Generate Cover Letter"):
            with output_placeholder.container():
                with st.spinner("Generating cover letter..."):
                    try:
                        crew = Crew(
                            agents=[resume_parser, jd_parser, cover_letter_generator],
                            tasks=[resume_task, jd_task, cover_letter_task],
                            name="Cover Letter Crew",
                            description="A team to generate a cover letter.",
                            verbose=True,
                            process=Process.sequential
                        )
                        result = crew.kickoff()
                        st.success("✅ Cover Letter Generation Complete")
                        try:
                            parsed_result = json.loads(result.raw)
                            st.markdown("### ✉️ Generated Cover Letter")
                            st.markdown(parsed_result["cover_letter"])
                        except Exception:
                            st.markdown("### ✉️ Generated Cover Letter")
                            st.markdown(result.raw)
                    except Exception as e:
                        st.error(f"⚠️ Cover letter generation failed: {str(e)}")
else:
    st.warning("Please provide both Resume and Job Description to proceed.")