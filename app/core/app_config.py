import os


class Configuration:
    def __init__(self):
        self.MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")


CONFIGURATION = Configuration()
