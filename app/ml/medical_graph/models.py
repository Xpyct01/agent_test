from pydantic import BaseModel, Field
from typing_extensions import Literal


class Route(BaseModel):
    step: Literal["CONTINUE", "END"] = Field(
        None, description="The next step in the routing process"
    )


class Symptoms(BaseModel):
    symptoms: list[str] = Field(
        None, description="The list of symptoms detected in user query"
    )


class Diagnoses(BaseModel):
    diagnoses: list[str] = Field(
        None, description="The list of diagnoses detected from user symptoms"
    )


class Questions(BaseModel):
    questions: list[str] = Field(
        None, description="The list of questions which could be used for better diagnosis detection"
    )


class Result(BaseModel):
    final_symptoms: list[str] = Field(
        None, description="The list of final symptoms obtained from user symptoms and user answers"
    )
    final_diagnoses: list[str] = Field(
        None, description="The list of final possible diagnoses obtained from previous diagnoses and user answers"
    )
    delay: int = Field(
        None, description="Possible delay of hospitalization in days (from 0 to 30), taking into account the patient's condition"
    )
    doctor: str = Field(
        None, description="The patient's doctor's specialization"
    )