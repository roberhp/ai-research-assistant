class FakeLLMProvider:
    def __init__(self, response: str = "Fake LLM response"):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response