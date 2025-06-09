from crewai import Agent, Task
class CoverLetterAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="Cover Letter Generator",
            backstory="I write cover letters tailored to job descriptions and candidate profiles.",
            goal="Generate persuasive cover letters that address gaps and emphasize fit."
        )

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        prompt = f"""
        Generate a professional cover letter that:
        - Highlights the candidate's strengths
        - Acknowledges and bridges skill or experience gaps
        - Aligns with the job description and tone

        Context:
        {context}
        """
        result = self.llm.call([{"role": "user", "content": prompt}])
        return json.dumps({"cover_letter": result.strip()})
