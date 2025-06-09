from crewai import Agent, Task
class ResumeEnhancerAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="Resume Enhancer",
            backstory="I suggest improvements to the resume to make it more relevant to the JD.",
            goal="Optimize resumes based on job descriptions."
        )

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        prompt = f"""
        Given this resume and job description, suggest improvements to the resume:
        - Highlight missing skills
        - Recommend better phrasing
        - Suggest added sections or content

        Context:
        {context}
        """
        result = self.llm.call([{"role": "user", "content": prompt}])
        return json.dumps({"resume_enhancement": result.strip()})
