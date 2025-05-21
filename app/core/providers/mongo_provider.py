from pymongo import MongoClient


class MongoProvider:
    def __init__(self, config):
        self._client = MongoClient(config.MONGO_CONNECTION_STRING)

    def get_client(self):
        return self._client
