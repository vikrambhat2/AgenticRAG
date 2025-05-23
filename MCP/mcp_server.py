import logging
from typing import Dict
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Init FastMCP
try:
    mcp = FastMCP("Resume JD Matcher")
except Exception as e:
    logging.error(f"Failed to initialize FastMCP: {e}")
    raise

# Set up Ollama model
try:
    model = OllamaLLM(model="llama3.2")
except Exception as e:
    logging.error(f"Failed to initialize OllamaLLM: {e}")
    raise

# -------------------------- #
# ------- TOOLS ------------ #
# -------------------------- #

@mcp.tool()
def parse_resume(resume_text: str) -> Dict:
    """
    Extract structured info from resume
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are a resume parser. Extract the following fields from the text:
        - Name
        - Education
        - Skills
        - Work experience (roles, companies, duration)

        Resume:
        {resume_text}
        """
    )
    chain = prompt | model
    result = chain.invoke({"resume_text": resume_text})
    # Handle LangChain output (e.g., AIMessage or list)
    if isinstance(result, list) and result:
        result = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])
    elif hasattr(result, 'content'):
        result = result.content  # Handle AIMessage
    return {"parsed_resume": str(result).strip()}

@mcp.tool()
def parse_jd(jd_text: str) -> Dict:
    """
    Extract structured info from Job Description
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are a JD parser. Extract:
        - Job title
        - Responsibilities
        - Required skills
        - Preferred skills

        Job Description:
        {jd_text}
        """
    )
    chain = prompt | model
    result = chain.invoke({"jd_text": jd_text})
    # Handle LangChain output
    if isinstance(result, list) and result:
        result = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])
    elif hasattr(result, 'content'):
        result = result.content
    return {"parsed_jd": str(result).strip()}

@mcp.tool()
def match_resume_to_jd(parsed_resume: str, parsed_jd: str) -> Dict:
    """
    Compare parsed resume and JD — list matches and mismatches
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Compare the following resume info to the job description info.

        Resume:
        {parsed_resume}

        Job Description:
        {parsed_jd}

        List skills or experiences that MATCH and those that are MISSING(present in Job Description but not in Resume ).
        Respond as JSON: {{"matched_skills": [], "missing_skills": [], "match_score": 0.0}}
        """
    )
    chain = prompt | model
    raw_output = chain.invoke({
        "parsed_resume": parsed_resume,
        "parsed_jd": parsed_jd
    })
    # Handle LangChain output
    if isinstance(raw_output, list) and raw_output:
        raw_output = raw_output[0].get('text', '') if isinstance(raw_output[0], dict) else str(raw_output[0])
    elif hasattr(raw_output, 'content'):
        raw_output = raw_output.content
    logging.info(f"[match_resume_to_jd] Raw model output:\n{raw_output}")
    try:
        parsed_output = json.loads(raw_output.strip())
        return {"match_resume": parsed_output}
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON output: {e}")
        return {"match_resume": raw_output.strip(), "error": "Invalid JSON"}

@mcp.tool()
def summarize_gap(parsed_resume: str, parsed_jd: str) -> Dict:
    """
    Generate a business-style gap analysis summary
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Generate a professional gap analysis summary comparing the resume and JD.

        Resume:
        {parsed_resume}

        JD:
        {parsed_jd}

        Highlight strengths, gaps, and whether the candidate is a good fit.
        """
    )
    chain = prompt | model
    result = chain.invoke({
        "parsed_resume": parsed_resume,
        "parsed_jd": parsed_jd
    })
    # Handle LangChain output
    if isinstance(result, list) and result:
        result = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])
    elif hasattr(result, 'content'):
        result = result.content
    return {"gap_summary": str(result).strip()}

# -------------------------- #
# ------- ENTRY ------------ #
# -------------------------- #

if __name__ == "__main__":
    try:
        logging.info("Launching Resume-JD Matcher MCP server with Ollama...")
        print("Starting mcp.run()...")
        mcp.run()
    except Exception as e:
        logging.error(f"Failed to run MCP server: {e}")
        raise