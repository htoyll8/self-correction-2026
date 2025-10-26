from openai import OpenAI


class Model:
    def __init__(self,
                 model_name="gpt-4o-mini",
                 temperature=0):
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def translate_to_logic(self, task_description, temperature=None):
        """
        Translate a natural-language task description into a First-Order Logic (FOL)
        specification using a stepwise prompting pipeline.
        """
        temperature = temperature if temperature is not None else self.temperature

        # Stage 1–6: one composite prompt (can later be broken into smaller sub-prompts)
        system_prompt = (
            "You are a logic-aware reasoning assistant that converts natural-language "
            "task descriptions into structured first-order logic (FOL). "
            "Follow these six steps, returning each in JSON:\n"
            "1. Identify claims and implications.\n"
            "2. Identify referring expressions (entities) and their relations.\n"
            "3. Identify predicates (properties/actions).\n"
            "4. Identify entailments or equivalences between predicates.\n"
            "5. Construct the First-Order Logic (FOL) formula.\n"
            "6. Optionally produce the negated formula and expected satisfiability.\n"
            "Return the final output as a JSON object with keys: step_1 … step_6."
        )

        # Run model
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task_description}"}
            ],
            temperature=temperature
        )

        try:
            fol_output = response.output_parsed or response.output_text
        except AttributeError:
            fol_output = response.output_text

        return fol_output

    def generate(self,
                 task_description,
                 n=1,
                 temperature=None):
        """
        Generate n initial programs (seeds) given a natural language 
        description.
        """
        temperature = (
            temperature if temperature is not None else self.temperature
        )

        responses = []

        for _ in range(n):
            response = self.client.responses.create(
                model=self.model_name,
                input=task_description,
                temperature=temperature,
            )
            code = response.output_text
            responses.append(code)

        return responses

    def generate_feedback(self,
                          task_description,
                          program,
                          temperature=None):
        """
        Ask the model to critique a program that failed its test cases.
        """
        temperature = temperature if temperature is not None else self.temperature

        prompt = (
            f"The following program did not pass its tests.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Program:\n{program}\n\n"
            f"Please explain what might be wrong and how to fix it."
        )

        response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=temperature,
            )

        return response.output_text

    def refine(self,
               task_description,
               program,
               feedback=None,
               temperature=None):
        """
        Ask the model to revise its program, either directly or using critique
        feedback.
        """
        temperature = temperature if temperature is not None else self.temperature

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

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=temperature,
        )

        return response.output_text
