class DocumentationAgent:

    SYSTEM_PROMPT = """
You are the Documentation Agent in a software engineering team.

Your job is to create clear documentation for an existing
software project.

Return ONLY the README.md contents.

The README should contain:

- Project title
- Description
- Features
- Installation instructions
- How to run the application
- API or usage documentation
- Example requests where appropriate

Rules:

- Document only functionality that actually exists.
- Inspect the provided source files carefully.
- Do not invent features.
- Return Markdown text only.
"""

    def __init__(self, llm):
        self.llm = llm

    def create(self, task, project_files):

        files_text = ""

        for filename, contents in project_files.items():
            files_text += f"""

FILE: {filename}

{contents}

"""

        return self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"""
ORIGINAL PROJECT REQUIREMENTS:

{task}

CURRENT PROJECT FILES:

{files_text}

Create README.md for this project.
""",
            max_new_tokens=1536,
            do_sample=False,
        )
