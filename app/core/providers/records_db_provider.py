class RecordsDBProvider:
    def __init__(self, mongo_client):
        self._db = mongo_client["TestAgent"]["records"]

    def get_db(self):
        return self._db
