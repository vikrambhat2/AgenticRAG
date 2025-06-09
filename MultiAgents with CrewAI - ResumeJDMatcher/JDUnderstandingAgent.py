
from crewai import Agent, Task
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
        return json.dumps({"job_description": structured_data.strip()})