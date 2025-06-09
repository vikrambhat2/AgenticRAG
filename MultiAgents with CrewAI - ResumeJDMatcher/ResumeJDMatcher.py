import streamlit as st
import json
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM, Process


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


# ----- Agent Definitions -----

class ResumeParsingAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="Resume Parser",
            backstory="I extract structured data from resumes with high accuracy.",
            goal="Parse resumes to extract skills, experience, education, certifications, and career gaps."
        )

    def execute_task(self, task: Task, context: list = None, tools: list = None):
        resume_content = task.description
        if not resume_content:
            raise ValueError("Resume content is required.")
        
        prompt = f"""
        You are a resume parser. Extract the following fields from the text:
        - Name
        - Education
        - Skills
        - Work experience (roles, companies, duration)

        Resume:
        {resume_content}
        """
        structured_data = self.llm.call([{"role": "user", "content": prompt}])
        return json.dumps({ "resume": structured_data.strip() })


class JDUnderstandingAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="JD Parser",
            backstory="I analyze job descriptions to extract key requirements.",
            goal="Extract mandatory/optional skills, seniority, and soft skills from job descriptions."
        )

    def execute_task(self, task: Task, context: list = None, tools: list = None):
        jd_content = task.description
        if not jd_content:
            raise ValueError("JD content is required.")
        
        prompt = f"""
        You are a JD parser. Extract:
        - Job title
        - Responsibilities
        - Required/Preferred skills

        Job Description:
        {jd_content}
        """
        structured_data = self.llm.call([{"role": "user", "content": prompt}])
        return json.dumps({ "job_description": structured_data.strip() })


class MatchingAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="Candidate-Role Matcher",
            backstory="I match candidates to roles based on skills and experience.",
            goal="Compute a match score between resume and job description."
        )

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        prompt = f"""

        Given the following resume and job_description in the context, identify:
        1. **Matched Skills** – Skills that are present in both the resume and the job description.
        2. **Missing Skills** – Skills that are required in the job description but not found in the resume.
        3. **Match Score** – A numerical score (0.0 to 1.0) representing the overall match quality.
        4. **Experience Match** – Whether the candidate's experience aligns with the job requirements.
        5. **Gaps** – Any significant gaps in skills or experience that may affect the candidate's fit.
        6. **Risk Flags** – Any potential red flags in the candidate's profile that may affect hiring decisions.
        7. **Soft Skills** – Any soft skills mentioned in the resume that match the job description.
        8. **Seniority Level** – Whether the candidate's experience matches the seniority level required by the job.
        context: {context}
        """
        result = self.llm.call([{"role": "user", "content": prompt}])
        return json.dumps({ "match_sumamry": result.strip() })


class RecommendationAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="Recommendation Agent",
            backstory="I create recommendations and cover letter for job seekers.",
            goal="Generate recommendations and cover letter based on the resume and job description."
        )

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        prompt = f"""
        Given the following resume and job_description in the context, write a personalized and professional short cover letter tailored to the specific job.

        The letter must:
        - Highlight the candidate's relevant skills, experience, and enthusiasm for the role.
        - Briefly and professionally address any skill or experience gaps, if present.
        - Be concise and suitable for immediate use—no additional explanations or commentary.

        context: {context}
        Only output the final cover letter.

        """
        return self.llm.call([{"role": "user", "content": prompt}])


# ----- Streamlit App -----

st.set_page_config(page_title="Resume-JD Matcher", layout="wide")
st.title("AI Resume vs JD Matching Tool")

# --- Input Columns ---
col1, col2 = st.columns(2)

with col1:
    st.header("Resume Input")
    resume_source = st.radio("Choose input method for Resume", ("Upload PDF", "Paste Text"))
    resume_text = ""
    if resume_source == "Upload PDF":
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_pdf")
        if resume_file:
            resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = st.text_area("Paste Resume Text", height=300)

    if resume_text:
        st.text_area("Resume Text Preview", resume_text, height=200)

with col2:
    st.header("Job Description Input")
    jd_source = st.radio("Choose input method for JD", ("Upload PDF", "Paste Text"))
    jd_text = ""
    if jd_source == "Upload PDF":
        jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd_pdf")
        if jd_file:
            jd_text = extract_text_from_pdf(jd_file)
    else:
        jd_text = st.text_area("Paste JD Text", height=300)

    if jd_text:
        st.text_area("JD Text Preview", jd_text, height=200)


# --- Run Matching ---
if resume_text and jd_text and st.button("Run Matching"):
    with st.spinner("Processing..."):

        # Initialize LLM
        try:
            llama_model = LLM(model="groq/meta-llama/llama-4-scout-17b-16e-instruct")

        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            st.stop()

        # Agents
        resume_parser = ResumeParsingAgent(llm=llama_model)
        jd_parser = JDUnderstandingAgent(llm=llama_model)
        matcher = MatchingAgent(llm=llama_model)
        recommender = RecommendationAgent(llm=llama_model)

        # Tasks
        resume_task = Task(
            description=resume_text,
            expected_output="Structured JSON with skills, experience, education, certifications, and career gaps.",
            agent=resume_parser
        )

        jd_task = Task(
            description=jd_text,
            expected_output="Structured JSON with mandatory/optional skills, seniority, and soft skills.",
            agent=jd_parser
        )

        matching_task = Task(
            description="Match resume to JD.",
            expected_output="JSON with match score, skill matches, experience match, and gaps.",
            agent=matcher,
            context=[resume_task, jd_task]
        )

        recommender_task = Task(
            description="Generate a final report.",
            expected_output="Readable text report with fit summary and recommendations.",
            agent=recommender,
            context=[resume_task, jd_task]
        )

        crew = Crew(
            agents=[resume_parser, jd_parser, matcher, recommender],
            tasks=[resume_task, jd_task, matching_task, recommender_task],
            name="Resume-JD Matcher Crew",
            description="A crew to parse resumes, analyze JDs, perform matching, and generate summaries.",
            verbose=True,
            process=Process.sequential
        )

        try:
            result = crew.kickoff()

            st.subheader("🔍 Matching Results")
            try:
                parsed_match = json.loads(matching_task.output.raw)
                st.markdown(parsed_match['match_sumamry'])
            except:
                st.text(matching_task.output.raw)

            st.subheader("📋 Cover Letter")
            st.markdown(result.raw)

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
