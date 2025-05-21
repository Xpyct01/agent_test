from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import Literal
from ml.state_schema import State


class ValidationRoute(BaseModel):
    result: Literal["TRUE", "FALSE"] = Field(
        None, description="Validation result"
    )


class QuestionsGraph:
    def __init__(self, model_provider):
        self.graph = self.create_graph()
        self.model = model_provider.llm
        self.validation_router = self.model.with_structured_output(ValidationRoute)

    def questions_node(self, state: State):
        current_question = state.get("current_question")
        questions = state.get("questions", [])
        if current_question is None and len(questions) > 0:
            current_question = questions.pop(0)
        else:
            current_question = ""

        answers = state.get("answers", [])
        clue = state.get("clue")
        if state.get('question_decision') == "TRUE":
            answers.append({"question": state["answered_question"], "answer": state["answer"]})

            if len(questions) > 0:
                current_question = questions.pop(0)
                clue = ""
            else:
                current_question = ""

        return {"current_question": current_question, "answers": answers, "questions": questions, "clue": clue}

    def question_route(self, state: State):
        return bool(state["current_question"])

    def human_node(self, state: State):
        if state.get('clue'):
            interrupt_value = state["clue"]
        else:
            interrupt_value = state["current_question"]
        answer = interrupt(interrupt_value)
        return {"answer": answer}

    def validation_node(self, state: State):
        decision_prompt_template = ChatPromptTemplate([
            ("system", "Route the input to TRUE if user provided answer to given question or FALSE if not."),
            ("user", "Given question: {question} \n\n User answer: {answer}")
        ])
        decision_prompt = decision_prompt_template.invoke({"question": state["current_question"],
                                                           "answer": state["answer"]})
        decision = self.validation_router.invoke(decision_prompt)

        clue = ""
        if decision.result == "FALSE":
            clue_prompt_template = ChatPromptTemplate([
                ("system", "give the user a hint that he needs to answer this question"),
                ("user", "Given question: {question}")
            ])
            clue_prompt = clue_prompt_template.invoke({"question": state["current_question"]})
            clue = self.model.invoke(clue_prompt)["content"]

        node_output = {
            "question_decision": decision.result, "answer": state["answer"],
            "answered_question": state["current_question"], "clue": clue
        }

        return node_output

    def validation_route(self, state: State):
        result_map = {"TRUE": True, "FALSE": False}
        return result_map[state["question_decision"]]

    def create_graph(self):
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node('questions_node', self.questions_node)
        graph_builder.add_node('human_node', self.human_node)
        graph_builder.add_node('validation_node', self.validation_node)

        graph_builder.add_edge(START, 'questions_node')
        graph_builder.add_conditional_edges(
            'questions_node',
            self.question_route,
            {
                True: 'human_node',
                False: END
            }
        )

        graph_builder.add_edge('human_node', 'validation_node')
        graph_builder.add_conditional_edges(
            'validation_node',
            self.validation_route,
            {
                True: 'questions_node',
                False: 'human_node',
            }
        )

        graph = graph_builder.compile()

        return graph
