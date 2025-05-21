from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from ml.medical_graph.models import Symptoms, Diagnoses, Questions, Result
from ml.medical_graph.prompts import summary_model_system_prompt, clarifying_model_system_prompt
from ml.state_schema import State
from ml.medical_graph.questions_graph import QuestionsGraph


class MedicalGraph:
    def __init__(self, model_provider):
        self.model = model_provider.llm
        self.symptoms_model = self.model.with_structured_output(Symptoms)
        self.diagnoses_model = self.model.with_structured_output(Diagnoses)
        self.questions_model = self.model.with_structured_output(Questions)
        self.summary_model = self.model.with_structured_output(Result)
        self.questions_graph = QuestionsGraph(model_provider).graph

        self.graph = self.create_graph()

    def symptom_extraction_node(self, state: State):
        symptoms_model_prompt_template = ChatPromptTemplate([
            ("system", "Your task is to detect symptoms in user query."),
            ("user", "Given query: {query}")
        ])
        symptoms_model_prompt = symptoms_model_prompt_template.invoke({"query": state["input"]})
        symptoms_model_output = self.symptoms_model.invoke(symptoms_model_prompt)
        symptoms = symptoms_model_output.symptoms
        return {"symptoms": symptoms}

    def pre_diagnosis_node(self, state: State):
        diagnosis_model_prompt_template = ChatPromptTemplate([
            ("system", "Your task is to detect a few possible diagnosis based on list of symptoms."),
            ("user", "Given symptoms: {symptoms}")
        ])
        diagnosis_model_prompt = diagnosis_model_prompt_template.invoke({"symptoms": state["symptoms"]})
        diagnosis_model_output = self.diagnoses_model.invoke(diagnosis_model_prompt)
        diagnoses = diagnosis_model_output.diagnoses
        return {"diagnoses": diagnoses}

    def clarifying_node(self, state: State):
        question_model_prompt_template = ChatPromptTemplate([
            ("system", clarifying_model_system_prompt),
            ("user", "Given symptoms: {symptoms} \n\n Possible diagnoses: {diagnoses}")
        ])
        question_model_prompt = question_model_prompt_template.invoke({"symptoms": state["symptoms"],
                                                                       "diagnoses": state["diagnoses"]})
        question_model_output = self.questions_model.invoke(question_model_prompt)
        questions = question_model_output.questions
        return {"questions": questions}

    def summary_node(self, state: State):
        summary_model_prompt_template = ChatPromptTemplate([
            ("system", summary_model_system_prompt),
            ("user", "Symptoms: {symptoms} \n\n Previous diagnoses: {diagnoses} \n\n User answers: {answers}")
        ])
        summary_model_prompt = summary_model_prompt_template.invoke({"symptoms": state["symptoms"],
                                                                     "diagnoses": state["diagnoses"],
                                                                     "answers": state["answers"]})
        summary_model_output = self.summary_model.invoke(summary_model_prompt)
        node_output = {"final_symptoms": summary_model_output.final_symptoms,
                       "final_diagnoses": summary_model_output.final_diagnoses,
                       "delay": summary_model_output.delay,
                       "doctor": summary_model_output.doctor}
        return node_output

    def create_graph(self):
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node("symptom_extraction_node", self.symptom_extraction_node)
        graph_builder.add_node("pre_diagnosis_node", self.pre_diagnosis_node)
        graph_builder.add_node("clarifying_node", self.clarifying_node)
        graph_builder.add_node("summary_node", self.summary_node)

        graph_builder.add_node("questions_graph", self.questions_graph)

        graph_builder.add_edge(START, "symptom_extraction_node")
        graph_builder.add_edge("symptom_extraction_node", "pre_diagnosis_node")
        graph_builder.add_edge("pre_diagnosis_node", "clarifying_node")
        graph_builder.add_edge("clarifying_node", "questions_graph")
        graph_builder.add_edge("questions_graph", "summary_node")

        graph = graph_builder.compile()

        return graph
