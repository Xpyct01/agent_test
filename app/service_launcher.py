import fastapi
from core.providers.model_provider import ModelProvider
from core.providers.mongo_provider import MongoProvider
from core.providers.memory_provider import MongoMemoryProvider
from core.providers.records_db_provider import RecordsDBProvider
from core.app_config import CONFIGURATION
from pydantic import BaseModel
from inference_graph import InferenceGraph


model_provider = ModelProvider()
mongo_client = MongoProvider(CONFIGURATION).get_client()
memory = MongoMemoryProvider(mongo_client).get_memory()
records_db = RecordsDBProvider(mongo_client).get_db()
graph = InferenceGraph(memory, records_db, model_provider)


class Query(BaseModel):
    user_id: int
    thread_id: int
    message: str


app = fastapi.FastAPI()


@app.get("/chat")
async def chat(query: Query):
    output = graph.inference(query)
    return {"output": output}
