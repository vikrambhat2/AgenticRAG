from crewai import Agent, Task
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
        1. Matched Skills
        2. Missing Skills
        3. Match Score (0.0 to 1.0)
        4. Experience Match
        5. Gaps
        6. Risk Flags
        7. Soft Skills
        8. Seniority Level

        Context:
        {context}
        """
        result = self.llm.call([{"role": "user", "content": prompt}])
        return json.dumps({"match_summary": result.strip()})
