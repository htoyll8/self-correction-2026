import re
import textwrap
import anthropic
from openai import OpenAI


class Model:
    def __init__(self,
                 model_name="gpt-4o-mini",
                 temperature=0):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        print(hasattr(self.client, "responses"))

        if self._is_claude():
            self.client = anthropic.Anthropic()
        else:
            self.client = OpenAI()

    def _is_claude(self):
        return self.model_name.lower().startswith("claude")

    def _sanitize_java_output(self, code: str) -> str:
        # Remove triple backticks and language tags
        code = re.sub(r"```(?:java)?", "", code, flags=re.IGNORECASE).strip()

        # Remove a lone leading "java" token
        code = re.sub(r"^\s*java\s*\n", "", code, flags=re.IGNORECASE)

        # Drop anything before the first import or class declaration
        match = re.search(r"(import\s+|class\s+)", code)
        if match:
            code = code[match.start():]

        return code.strip()

    def _claude_generate(self, prompt, max_tokens=2048, temperature=None):
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        message = self.client.messages.create(**kwargs)
        return message.content[0].text

    def generate(
            self,
            task_description,
            n=1,
            temperature=None,
            language="python"):
        """
        Generate n initial programs (seeds) given a natural language description.
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        responses = []

        for _ in range(n):
            if language == "java":
                prompt = (
                    task_description
                    + "\n\n"
                    + "Write a complete Java class named `Solution` that implements the method above.\n"
                    + "Requirements:\n"
                    + "• Include `import java.util.*;` and `import java.lang.*;` at the top.\n"
                    + "• The class must be named exactly `Solution`.\n"
                    + "• Do NOT include a `main` method or any other class.\n"
                    + "\n"
                    + "Provide the code in a single ```java block:\n"
                    + "```java\n"
                    + "import java.util.*;\n"
                    + "import java.lang.*;\n"
                    + "\n"
                    + "class Solution {\n"
                    + "    // your method here\n"
                    + "}\n"
                    + "```\n\n"
                )
            elif language == "python":
                prompt = (
                    task_description +
                    "Please provide Python code wrapped in triple backticks like:\n"
                    "```python\n"
                    "# your code here\n"
                    "```\n\n"
                )
            else:
                lang_map = {
                    "cpp": ("C++",          "cpp",        "// your code here"),
                    "go":  ("Go",           "go",         "// your code here"),
                    "js":  ("JavaScript",   "javascript", "// your code here"),
                }
                lang_name, lang_tag, lang_comment = lang_map.get(language, ("Python", "python", "# your code here"))
                prompt = (
                    task_description +
                    f"Please provide {lang_name} code wrapped in triple backticks like:\n"
                    f"```{lang_tag}\n"
                    f"{lang_comment}\n"
                    "```\n\n"
                )

            # request = dict(
            #     model=self.model_name,
            #     input=(
            #         task_description +
            #         "Please provide Java code wrapped in triple backticks like:\n"
            #         "```java\n"
            #         "// your code here\n"
            #         "```\n\n"
            #     ),
            # )

            # prompt = (
            #     task_description +
            #     "Please provide Java code wrapped in triple backticks like:\n"
            #     "```java\n"
            #     "// your code here\n"
            #     "```\n\n"
            # )

            # prompt = (
            #     task_description
            #     + "\n\n"
            #     + "IMPORTANT JAVA OUTPUT REQUIREMENTS:\n"
            #     + "• You are writing a complete Java class named `Solution` for this task.\n"
            #     + "• The class must contain a single method with the signature described in the task "
            #     "(e.g., `public boolean hasCloseElements(List<Double> numbers, double threshold)`).\n"
            #     + "• Include all required imports (`import java.util.*; import java.lang.*;`) inside the class.\n"
            #     + "• The method should implement the logic to solve the task.\n"
            #     + "• Do NOT include any extra classes, code fences, or explanations.\n"
            #     + "\n"
            #     + "Example structure:\n"
            #     + "import java.util.*;\n"
            #     + "import java.lang.*;\n"
            #     + "\n"
            #     + "class Solution {\n"
            #     + "    public boolean exampleMethod(List<Double> numbers, double threshold) {\n"
            #     + "        // implementation here\n"
            #     + "    }\n"
            #     + "}\n"
            #     + "\n"
            #     + "Return ONLY the complete Java class code. Nothing else."
            # )

            # prompt = (
            #     task_description
            #     +
            #     "\n\n"
            #     "IMPORTANT REQUIREMENTS FOR PYTHON SOLUTION:\n"
            #     "1. The solution MUST NOT define a function such as `def solve()` or `def main()`.\n"
            #     "2. The solution MUST be a top-level script that executes immediately when run.\n"
            #     "3. Programs MUST read input **exactly** as shown in the examples. This problem uses\n"
            #     "   standard input (stdin). The program MUST:\n"
            #     "   • Read all input in the same structure and ordering as in the examples.\n"
            #     "   • Handle multi-line input exactly as given.\n"
            #     "   • Produce output in exactly the same format as the example output.\n"
            #     "\n"
            #     "Your code must begin with something like the following (this is ONLY an example):\n"
            #     "```python\n"
            #     "import sys\n"
            #     "data = sys.stdin.read().strip().split()\n"
            #     "```\n"
            #     "or:\n"
            #     "```python\n"
            #     "n = int(input().strip())\n"
            #     "```\n"
            #     "depending on the input format.\n"
            #     "\n"
            #     "Please provide ONLY executable Python code wrapped in triple backticks like:\n"
            #     "```python\n"
            #     "# your code here\n"
            #     "```\n"
            #     "Do not include any explanation or text outside the code block.\n"
            # )

            # 🔀 Route by provider
            if self._is_claude():
                code = self._claude_generate(
                    prompt,
                    temperature=temperature
                )
            else:
                request = {
                    "model": self.model_name,
                    "input": prompt,
                }

                # GPT-5 does not support temperature
                if temperature is not None and "gpt-5" not in self.model_name.lower():
                    request["temperature"] = temperature

                response = self.client.responses.create(**request)
                code = response.output_text

            # 🔧 NEW: sanitize Java output
            if language == "java":
                code = self._sanitize_java_output(code)

            responses.append(code)

        return responses

    def generate_feedback(self, task_description, program_or_context, temperature=None):
        """
        Ask the model to critique a failed attempt (program only or with history context).
        Automatically adjusts the prompt if history context is included.
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        has_history = "Summary of previous attempts:" in program_or_context
        section_label = (
            "Context (previous attempts and current program)"
            if has_history else
            "Program to Critique"
        )

        # Build prompt
        prompt = (
            f"The following attempt did not pass all of its tests.\n\n"
            f"Please explain what might be wrong.\n\n"
            f"Task:\n{task_description}\n\n"
            f"{section_label}:\n{program_or_context}\n\n"
        )

        # prompt = (
        #     f"The following attempt did not pass all of its tests.\n\n"
        #     f"Please explain what might be wrong.\n\n"
        #     f"Task:\n{task_description}\n\n"
        #     f"{section_label}:\n{program_or_context}\n\n"
        #     f"IMPORTANT JAVA OUTPUT REQUIREMENTS:\n"
        #     f"• You are writing a complete Java class named `Solution` for this task.\n"
        #     f"• The class must contain a single method with the signature described in the task "
        #     f"(e.g., `public boolean hasCloseElements(List<Double> numbers, double threshold)`).\n"
        #     f"• Include all required imports (`import java.util.*; import java.lang.*;`) inside the class.\n"
        #     f"• The method should implement the logic to solve the task.\n"
        #     f"• Do NOT include any extra classes, code fences, or explanations.\n\n"
        #     f"Example structure:\n"
        #     f"import java.util.*;\n"
        #     f"import java.lang.*;\n\n"
        #     f"class Solution {{\n"
        #     f"    public boolean exampleMethod(List<Double> numbers, double threshold) {{\n"
        #     f"        // implementation here\n"
        #     f"    }}\n"
        #     f"}}\n\n"
        #     f"Return ONLY the complete Java class code. Nothing else."
        # )

        # prompt = textwrap.dedent(f"""
        # The following attempt did not pass all of its tests.

        # Please explain what might be wrong with the program, including issues in logic,
        # input handling, and output formatting.

        # IMPORTANT REQUIREMENTS FOR PYTHON SOLUTIONS:
        # 1. The solution MUST NOT define a function such as `def solve()` or `def main()`.
        # 2. The solution MUST be a top-level script that executes immediately when run.
        # 3. Programs MUST read input **exactly** as shown in the examples. This problem uses
        # standard input (stdin). The program MUST:
        # • Read all input in the same structure and ordering as in the examples.
        # • Handle multi-line input exactly as given.
        # • Produce output in exactly the same format as the example output.

        # ====================
        # TASK DESCRIPTION
        # ====================
        # {task_description}

        # ====================
        # {section_label.upper()}
        # ====================
        # {program_or_context}
        # """)

        print(f"DEBUG [model.py]: Feedback prompt: {prompt}", flush=True)

        # 🔀 Route by provider
        if self._is_claude():
            return self._claude_generate(
                prompt,
                temperature=temperature
            )

        # ---- OpenAI path (unchanged behavior) ----
        request = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }

        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature

        response = self.client.responses.create(**request)
        return response.output_text

    def generate_antiunified_history(self, trajectory, temperature=None):
        """
        Ask the model to perform *anti-unification* over all previous program attempts.
        Produces a single abstracted form capturing their shared structure without truncation.
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        attempts = trajectory.get("refinement_attempts", [])
        if not attempts:
            return ""

        # 🧩 Include all prior programs in full — no truncation, no feedback.
        program_snippets = []
        for r in attempts:
            program_snippets.append(
                f"### Attempt {r['attempt']} (pass rate: {r['pass_fraction']*100:.1f}%)\n"
                f"{r['program']}\n"
            )

        programs_text = "\n\n".join(program_snippets)

        # 🧠 Construct a precise but lean prompt.
        prompt = (
            "Given several program variants that attempt to solve the same problem, "
            "derive a single generalized form that captures their shared structure.\n\n"
            "Rules for generalization:\n"
            "- Preserve common syntax and control flow.\n"
            "- Replace differing expressions, constants, or statements with placeholders "
            "such as <EXPR>, <VAR>, or <COND>.\n"
            "- Do not paraphrase or summarize the code — output a concrete unified program skeleton.\n\n"
            "Here are the previous attempts:\n\n"
            f"{programs_text}\n\n"
            "Produce the anti-unified abstraction below:\n"
        )

        # Send prompt to model
        request = {
            "model": self.model_name,
            "input": [{"role": "user", "content": prompt}],
        }

        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature

        response = self.client.responses.create(**request)
        return response.output_text

    def refine(self,
               task_description,
               program,
               feedback=None,
               temperature=None):
        """
        Ask the model to revise its program, either directly or using critique feedback.
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        if feedback:
            prompt = (
                f"Task:\n{task_description}\n\n"
                f"Current Program:\n{program}\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Revise the program to address the feedback. "
                f"Only return the corrected code."
            )
        else:
            prompt = (
                f"Task:\n{task_description}\n\n"
                f"Current Program:\n{program}\n\n"
                f"Revise and improve the program to make it pass all tests."
            )

        # 🔀 Claude path
        if self._is_claude():
            return self._claude_generate(
                prompt,
                temperature=temperature
            )

        # ---- OpenAI path (unchanged behavior) ----
        request = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": prompt
                },
            ],
        }

        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature

        response = self.client.responses.create(**request)
        return response.output_text
