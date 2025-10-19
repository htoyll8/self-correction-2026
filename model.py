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
