import folktexts
import pdb
import pandas as pd

from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset
import os
import litellm
litellm._turn_on_debug()

TASK_DESCRIPTION = """\
This dataset contains diagnostic health information for female patients of Pima Indian heritage, all at least 21 years old. \
Each record includes medical predictor variables such as glucose level, BMI, age, insulin level, and more. \
The goal is to classify whether a patient has diabetes or not, based on these medical indicators. \
Use the data provided to make an informed diagnosis prediction.
"""
# ,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
pregnancies_col = ColumnToText(
    "Pregnancies",
    short_description="Number of pregnancies the patient has had",
    value_map=lambda x: x  # Numeric feature
)

glucose_col = ColumnToText(
    "Glucose",
    short_description="Plasma glucose concentration",
    value_map=lambda x: x
)

blood_pressure_col = ColumnToText(
    "BloodPressure",
    short_description="Diastolic blood pressure (mm Hg)",
    value_map=lambda x: x
)

skin_thickness_col = ColumnToText(
    "SkinThickness",
    short_description="Triceps skin fold thickness (mm)",
    value_map=lambda x: x
)

insulin_col = ColumnToText(
    "Insulin",
    short_description="2-hour serum insulin (mu U/ml)",
    value_map=lambda x: x
)

bmi_col = ColumnToText(
    "BMI",
    short_description="Body Mass Index (weight in kg / height in m²)",
    value_map=lambda x: x
)

diabetes_pedigree_col = ColumnToText(
    "DiabetesPedigreeFunction",
    short_description="Diabetes pedigree function (likelihood of diabetes based on family history)",
    value_map=lambda x: x
)

age_col = ColumnToText(
    "Age",
    short_description="Age of the patient (years)",
    value_map=lambda x: x
)

diabetes_col = ColumnToText(
    "Outcome",
    short_description="diabetes diagnosis",
    value_map={
        1: 'Yes',
        0: 'No'
    }
)

reentry_numeric_qa = DirectNumericQA(
    column='Outcome',
    text=(
        "Does this female patient have diabetes?"
    ),
)

reentry_qa = MultipleChoiceQA(
    column='Outcome',
    text="Does this female patient have diabetes?",
    choices=(
        Choice("Yes, she has diabetes", 1),
        Choice("No, she doesn't have diabetes", 0),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["Outcome"]
discretize_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

reentry_task = TaskMetadata(
    name="Indian Diabetes prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='Outcome',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

data = pd.read_csv("../data/IndianDiabetes.csv", index_col=0)
num_data = len(data)
# we want to sample 10k
# subsampling = 5000 / num_data

reentry_dataset = Dataset(
    data=data,
    task=reentry_task,
    test_size=0.95,
    val_size=0,
    # subsampling=subsampling,   # NOTE: Optional, for faster but noisier results!
)

all_tasks = {
    "reentry": [reentry_task, reentry_dataset]
}


model_name = "openai/gpt-4o-mini"
import os
import json
os.environ["OPENAI_API_KEY"] = json.loads("secrets.txt")["open_ai_key"]

for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task, custom_prompt_prefix=TASK_DESCRIPTION)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "IndianDiabetes"
    bench.run(results_root_dir=RESULTS_DIR)
