clarifying_model_system_prompt = """
Your task is to make up to 5 questions to user to better diagnosis recognition. 
You will receive detected symptoms in user query and possible diagnoses.
"""


summary_model_system_prompt = """
Your task is to extract final symptoms, final diagnosis, delay and doctor's specialization based on symptoms from user query and previous diagnoses. 
Delay is possible postponement of hospitalization in days (from 0 to 30), taking into account the patient's condition."
"""
