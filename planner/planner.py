import json


class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, project_request):
        prompt = f"""
You are the Planner for a multi-agent software development team.

The available roles are exactly:

- developer
- tester
- documentation
- reviewer

Your job is to convert the project request into a dependency-aware task plan.

PROJECT REQUEST:
{project_request}

Rules:

1. Break the project into concrete executable tasks.

2. Each task must be assigned to exactly one available role.

3. Use dependencies only when one task genuinely cannot begin before another.

4. Maximize parallelism. If two tasks can reasonably be performed independently,
   do not add a dependency between them.

5. Prefer a shallow dependency graph over a long serial chain.

6. Never create a dependency solely because two tasks are conceptually related.

7. Prefer sibling tasks that share the same prerequisite instead of chaining
   those tasks together.

   Example:
   If task A and task B can both start after task X, use:
       A depends on X
       B depends on X

   Do NOT use:
       A depends on X
       B depends on A

   unless B truly requires A's output.

8. Avoid dependency chains longer than necessary.

9. For small and medium projects, target a dependency-graph depth of roughly
   4 to 8 whenever the real workflow allows it.

10. Tester tasks should depend on the implementation they test, not on
    unrelated tester tasks.

11. Documentation tasks should depend only on the implementation or behavior
    they document, not on unrelated testing or review tasks.

12. Independent implementation components should be represented as parallel
    sibling tasks whenever possible.

6. For small or medium projects, prefer approximately 6 to 15 total tasks.
   Do not create one task for every tiny method, endpoint, test case, or
   documentation section unless that separation is necessary.

7. Group closely related implementation work into coherent developer tasks.

8. Group related tests into one or a small number of tester tasks.

9. Group documentation work into one or a small number of documentation tasks.

10. Prefer ONE final reviewer task for the whole project.
    Add intermediate reviewer tasks only when there is a strong technical reason.

11. Do not create one reviewer task per feature.

12. Documentation does not need to depend on testing unless test results are
    specifically required to write the documentation.

13. Developer tasks should implement or repair software.

14. Tester tasks should verify functionality, integration behavior, and edge cases.

15. Documentation tasks should document setup, usage, APIs, and important behavior.

16. Reviewer tasks should perform final quality review.

17. Do not invent requirements that are not implied by the project request.

18. Every dependency must reference a valid task that appears EARLIER in the plan.

19. Do not create dependencies on unrelated tasks.

20. The final review task should depend on the major implementation, testing,
    and documentation work required for project completion.

21. Return ONLY valid JSON.

22. Do not use markdown.

Return exactly this structure:

{{
  "project": "short project name",
  "tasks": [
    {{
      "id": "T1",
      "role": "developer",
      "description": "specific task",
      "depends_on": []
    }}
  ]
}}
"""

        result = self.llm.generate(
            "You are a software project planning agent. "
            "Return only valid JSON matching the requested schema.",
            prompt,
            max_new_tokens=2048,
            do_sample=False,
        )

        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)

        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]

        if text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        return json.loads(text)
