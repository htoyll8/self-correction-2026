from openai import OpenAI


class Model:
    def __init__(self,
                 model_name="gpt-4o-mini",
                 temperature=0):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def generate(self,
                 task_description,
                 n=1,
                 temperature=None):
        """
        Generate n initial programs (seeds) given a natural language description.
        """
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        responses = []

        for _ in range(n):
            # Build base request
            request = dict(
                model=self.model_name,
                input=task_description,
            )

            # Add temperature only if supported
            if temperature is not None and "gpt-5" not in self.model_name.lower():
                request["temperature"] = temperature

            # Send request safely
            response = self.client.responses.create(**request)
            code = response.output_text
            responses.append(code)

        return responses

    def generate_feedback(self, task_description, program_or_context, temperature=None):
        """
        Ask the model to critique a failed attempt (program only or with history context).
        Automatically adjusts the prompt if history context is included.
        """
        # Resolve temperature
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        # Detect whether this input includes history context
        has_history = "Summary of previous attempts:" in program_or_context

        # Dynamically adjust the label
        section_label = "Context (previous attempts and current program)" if has_history else "Program"

        # Build prompt
        prompt = (
            f"The following attempt did not pass all of its tests.\n\n"
            f"Task:\n{task_description}\n\n"
            f"{section_label}:\n{program_or_context}\n\n"
            f"Please explain what might be wrong and how to fix it."
        )

        # Construct request
        request = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": prompt
                },
            ],
        }

        # Only include temperature for non–GPT-5 models
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature

        # Send request
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
        Removes temperature if model is GPT-5 (which does not support it).
        """
        # Use provided temperature or fall back to default
        temperature = temperature if temperature is not None else getattr(self, "temperature", None)

        # Build the base prompt
        if feedback:
            prompt = (
                f"Task:\n{task_description}\n\n"
                f"Current Program:\n{program}\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Revise the program to address the feedback. Only return the corrected code."
            )
        else:
            prompt = (
                f"Task:\n{task_description}\n\n"
                f"Current Program:\n{program}\n\n"
                f"Revise and improve the program to make it pass all tests."
            )

        # Construct request dictionary
        request = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": prompt
                },
            ],
        }

        # Add temperature only if the model supports it
        if temperature is not None and "gpt-5" not in self.model_name.lower():
            request["temperature"] = temperature

        # Send request
        response = self.client.responses.create(**request)
        return response.output_text
