from crewai import Agent, Task
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
        return json.dumps({"resume": structured_data.strip()})
