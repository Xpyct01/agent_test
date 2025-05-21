from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    input: str
    chat_decision: str
    symptoms: list
    diagnoses: list
    question_decision: str
    questions: list
    current_question: str
    answered_question: str
    answers: dict
    answer: str
    clue: str
    final_symptoms: list
    final_diagnoses: list
    delay: int
    doctor: str
