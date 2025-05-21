from langchain.chat_models import init_chat_model


llm = init_chat_model("gpt-4o-mini", model_provider="openai")


class ModelProvider:
    def __init__(self):
        self.llm = llm
