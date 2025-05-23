from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import Literal
from ml.medical_graph.core import MedicalGraph
from ml.state_schema import State
import datetime


class ChatRoute(BaseModel):
    step: Literal["MEDICAL", "CHATBOT"] = Field(
        None, description="The next step in the routing process"
    )


class InferenceGraph:
    def __init__(self, memory, records_db, model_provider):
        self.model = model_provider.llm
        self.records_db = records_db
        self.router = self.model.with_structured_output(ChatRoute)
        self.medical_graph = MedicalGraph(model_provider).graph
        self.graph = self.create_graph(memory)

    def llm_call_router(self, state: State):
        prompt = ("Route the input to MEDICAL if the user complained of feeling unwell, named symptoms, etc., "
                  "otherwise CHATBOT.")
        decision = self.router.invoke(
            [
                SystemMessage(
                    content=prompt
                ),
                HumanMessage(content=state["input"]),
            ]
        )
        return {"chat_decision": decision.step}

    def chatbot(self, state: State):
        return {"messages": [self.model.invoke(state["messages"])]}

    def route_decision(self, state: State):
        return state["chat_decision"]

    def create_graph(self, memory):
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node("llm_call_router", self.llm_call_router)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("medial_graph", self.medical_graph)

        graph_builder.add_edge(START, "llm_call_router")
        graph_builder.add_conditional_edges(
            "llm_call_router",
            self.route_decision,
            {
                "CHATBOT": "chatbot",
                "MEDICAL": "medial_graph"
            },
        )

        graph = graph_builder.compile(checkpointer=memory)

        return graph

    def get_record(self, user_id, output_values):
        delay = output_values["delay"]
        result_date = datetime.datetime.now() + datetime.timedelta(days=delay)
        result_date = result_date.strftime('%d.%m.%Y')
        record = {
            "user_id": user_id, "doctor": output_values["doctor"], "date": result_date,
            "symptoms": output_values["final_symptoms"], "diagnoses": output_values["final_diagnoses"]
        }
        return record

    def insert_record(self, record):
        self.records_db.insert_one(record)

    def inference(self, query):
        config = {"configurable": {"thread_id": query.session_id}}
        input_state = self.graph.get_state(config)

        if len(input_state.interrupts) > 0:
            graph_output = self.graph.invoke(Command(resume=query.message), config=config)
        else:
            input_messages = [HumanMessage(query.message)]
            graph_output = self.graph.invoke({"input": query.message, "messages": input_messages}, config)

        output_state = self.graph.get_state(config)
        if '__interrupt__' in graph_output:
            output = graph_output['__interrupt__'][0].value
        elif output_state.values.get('delay', None) is not None:
            record = self.get_record(query.user_id, output_state.values)
            self.insert_record(record)
            output = f"You have an appointment with a {record['doctor']} on {record['date']}"
            self.graph.update_state(config, {'delay': None})
        else:
            output = graph_output['messages'][-1].content

        return output
