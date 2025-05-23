import streamlit as st
import asyncio
from PyPDF2 import PdfReader
from fastmcp import Client
import json

# MCP endpoint
MCP_URL = "server.py"  # Replace with actual running MCP server address


# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


# Async wrapper to call MCP tool
async def call_mcp_tool(tool_name, inputs):
    async with Client(MCP_URL) as client:
        response = await client.call_tool(tool_name, inputs)
        return response


# Wrapper for Streamlit to run async function
def run_async_tool(tool_name, inputs):
    return asyncio.run(call_mcp_tool(tool_name, inputs))


# UI Setup
st.set_page_config(page_title="Smart Resume Matcher", layout="wide")
st.title("📄 Smart Resume Analyzer (MCP)")

col1, col2 = st.columns(2)

# ---- Resume Input Section ----
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

# ---- JD Input Section ----
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

# ---- Main Action ----
if st.button("Match Resume to JD"):
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please provide both Resume and Job Description content.")
    else:
        with st.spinner("🔍 Parsing Resume..."):
            parsed_resume = run_async_tool("parse_resume", {"resume_text": resume_text})
            parsed_resume_content = json.loads(parsed_resume[0].text)

        with st.spinner("🔍 Parsing JD..."):
            parsed_jd = run_async_tool("parse_jd", {"jd_text": jd_text})
            parsed_jd_content = json.loads(parsed_jd[0].text)

        with st.spinner("🤝 Matching Resume to JD..."):
            match_result = run_async_tool("match_resume_to_jd", {
                "parsed_resume": parsed_resume_content["parsed_resume"],
                "parsed_jd": parsed_jd_content["parsed_jd"]
            })

        with st.spinner("📉 Summarizing Gaps..."):
            gap_summary = run_async_tool("summarize_gap", {
                "parsed_resume": parsed_resume_content["parsed_resume"],
                "parsed_jd": parsed_jd_content["parsed_jd"]
            })

        # ---- Display Results ----
        st.subheader("✅ Match Result")
        try:
            st.json(json.loads(match_result[0].text))
        except Exception as e:
            st.markdown(match_result[0].text)

        st.subheader("📊 Gap Summary")
        try:
            gap_summary_json = json.loads(gap_summary[0].text)
            st.markdown(gap_summary_json.get("gap_summary", "No summary returned."))
        except Exception as e:
            st.error(f"Error parsing gap summary: {e}")
