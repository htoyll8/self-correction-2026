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
        pass

    def refine(self,
               task_description,
               program,
               feedback=None,
               temperature=None):
        """
        Ask the model to revise its program, either directly or using critique
        feedback.
        """
        pass
