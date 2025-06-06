import folktexts
import pandas as pd

from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset
import os


TASK_DESCRIPTION = """\
The following data corresponds to ICU patient records in the US, someone whom are in danger of contracting sepsis. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

"""COLUMNS"""
hour = ColumnToText(
    "Hour",
    short_description="number of hours spent in the ICU",
    value_map=lambda x: f"{int(x)} hours",
)

hr_col = ColumnToText(
    "HR",
    short_description="heart rate in BPM",
    value_map=lambda x: f"{float(x)} BPM",
)

o2_col = ColumnToText(
    "O2Sat",
    short_description="O2Sat Pulse oximetry",
    value_map=lambda x: f"{int(x)} percent",
)

temp_col = ColumnToText(
    "Temp",
    short_description="Temperature",
    value_map=lambda x: f"{float(x)} degrees F"
)

sbp_col = ColumnToText(
    "SBP",
    short_description="Systolic blood pressure",
    value_map=lambda x: f"{float(x)} mm Hg"
)

map_col = ColumnToText(
    "MAP",
    short_description="mean arterial pressure",
    value_map=lambda x: f"{(float(x))} mm Hg"
)

dbp_col = ColumnToText(
    "DBP",
    short_description="Diastolic blood pressure",
    value_map=lambda x: f"{float(x)} mm Hg"
)

resp_col = ColumnToText(
    "Resp",
    short_description="respiration rate",
    value_map=lambda x: f"{float(x)} breaths per minute"
)

etco2_col = ColumnToText(
    "EtCO2",
    short_description="EtCO2",
    value_map=lambda x: f"{float(x)} mm Hg"
)

excess_col = ColumnToText(
    "BaseExcess",
    short_description="excess bicarbonate",
    value_map=lambda x: f"{float(x)} mmol / L"
)

hco3_col = ColumnToText(
    "HCO3",
    short_description="HCO3",
    value_map=lambda x: f"{float(x)} mmol / L"
)

fio2_col = ColumnToText(
    "FiO2",
    short_description="FiO2",
    value_map=lambda x: f"{float(x)} percent"
)

ph_col = ColumnToText(
    "pH",
    short_description="pH",
    value_map=lambda x: f"{float(x)}"
)

paco2_col = ColumnToText(
    "PaCO2",
    short_description="PaCO2",
    value_map=lambda x: f"{float(x)} mm Hg"
)

sao2_col = ColumnToText(
    "SaO2",
    short_description="SaO2",
    value_map=lambda x: f"{float(x)} percent"
)

ast_col = ColumnToText(
    "AST",
    short_description="aspratate transaminase",
    value_map=lambda x: f"{float(x)} IU / L"
)

bun_col = ColumnToText(
    "BUN",
    short_description="blood urea nitrogen",
    value_map=lambda x: f"{float(x)} mg / dL"
)

Alkalinephos_col = ColumnToText(
    "Alkalinephos",
    short_description="Alkalinephos",
    value_map=lambda x: f"{float(x)} IU / L"
)

calcium_col = ColumnToText(
    "Calcium",
    short_description="calcium",
    value_map=lambda x: f"{float(x)} mg / dL"
)

chloride_col = ColumnToText(
    "Chloride",
    short_description="chloride",
    value_map=lambda x: f"{float(x)} mmol / L"
)

creatinine_col = ColumnToText(
    "Creatinine",
    short_description="creatinine",
    value_map=lambda x: f"{float(x)} mg / dL"
)

Bilirubin_direct_col = ColumnToText(
    "Bilirubin_direct",
    short_description="bilirubin",
    value_map=lambda x: f"{float(x)} mg / dL"
)

troponin_col = ColumnToText(
    "TroponinI",
    short_description="troponin",
    value_map=lambda x: f"{float(x)} ng / mL"
)

hematocrit_col = ColumnToText(
    "Hct",
    short_description="hematocrit",
    value_map=lambda x: f"{float(x)} percent"
)

hemoglobin_col = ColumnToText(
    "Hgb",
    short_description="hemoglobin",
    value_map=lambda x: f"{float(x)} g / dL"
)

thromboplastin_col = ColumnToText(
    "PTT",
    short_description="partial thromboplastin time",
    value_map=lambda x: f"{float(x)} seconds"
)

wbc_col = ColumnToText(
    "WBC",
    short_description="white blood cell count",
    value_map=lambda x: f"{float(x)} thousand per microliter"
)

fibrogen_col = ColumnToText(
    "Fibrinogen",
    short_description="Fibrinogen",
    value_map=lambda x: f"{float(x)} mg / dL"
)

platelets_col = ColumnToText(
    "Platelets",
    short_description="platelet count",
    value_map=lambda x: f"{float(x)} thousand per microliter"
)

iculos_col = ColumnToText(
    "ICULOS",
    short_description="hours since ICU admit",
    value_map=lambda x: f"{x} hours"
)

label_col = ColumnToText(
    "SepsisLabel",
    short_description="whether patient contracted sepsis within 6 hours",
    value_map={
        0: "no",
        1: "yes"
    }
)


"""QUESTIONS"""


reentry_numeric_qa = DirectNumericQA(
    column='SepsisLabel',
    text=(
        "Did the patient contract sepsis within 6 hours?"
    ),
)


reentry_qa = MultipleChoiceQA(
    column='SepsisLabel',
    text="Did the patient contract sepsis within 6 hours?",
    choices=(
        Choice("Yes, they did", 1),
        Choice("No, they did not", 0),
    ),
)


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


# all_outcomes = ['JAIL', 'FOUR_ER', 'EMERG_SHLTR', 'MHIP']
all_outcomes = ["SepsisLabel"]

reentry_task = TaskMetadata(
    name="sepsis prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='SepsisLabel',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)



data = pd.read_csv("data/sepsis.csv")
num_data = len(data)
# we want to sample 5k
subsampling = 5 / num_data

reentry_dataset = Dataset(
    data=data,
    task=reentry_task,
    test_size=0.95,
    val_size=0,
    subsampling=subsampling,   # NOTE: Optional, for faster but noisier results!
)




all_tasks = {
    "reentry": [reentry_task, reentry_dataset]
}


model_name = "openai/gpt-4o-mini"
import os
import json
with open("secrets.json", "r") as handle:
    os.environ["OPENAI_API_KEY"] = json.load(handle)["open_ai_key"]
    
for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task, custom_prompt_prefix=TASK_DESCRIPTION)  
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "sepsis"
    bench.run(results_root_dir=RESULTS_DIR)

