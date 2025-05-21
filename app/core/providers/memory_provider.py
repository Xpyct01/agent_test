from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import InMemorySaver


class MongoMemoryProvider:
    def __init__(self, mongo_client):
        self._memory = MongoDBSaver(mongo_client)

    def get_memory(self):
        return self._memory


class LocalMemoryProvider:
    def __init__(self):
        self._memory = InMemorySaver()

    def get_memory(self):
        return self._memory
