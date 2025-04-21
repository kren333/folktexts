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

data = pd.read_csv("data/bank_full_binary.csv")
num_data = len(data)

TASK_DESCRIPTION = """
The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact with the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.\

Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

"""COLUMNS"""
age = ColumnToText(
    "age",
    short_description="years of age",
    value_map=lambda x: f"{x} years",
)

job = ColumnToText(
    "job",
    short_description="job type",
    value_map=lambda x: f"{x}",
)

marital = ColumnToText(
    "marital",
    short_description="marital status",
    value_map=lambda x: f"{x}",
)

education = ColumnToText(
    "education",
    short_description="highest level of education",
    value_map=lambda x: f"{x}",
)

default = ColumnToText(
    "default",
    short_description="history of prior defaults",
    value_map=lambda x: f"{x}",
)

balance = ColumnToText(
    "balance",
    short_description="average yearly balance (euros)",
    value_map=lambda x: f"{x} euros",
)

housing = ColumnToText(
    "housing",
    short_description="existing of housing loan",
    value_map=lambda x: f"{x}",
)

loan = ColumnToText(
    "loan",
    short_description="existing of personal loan",
    value_map=lambda x: f"{x}",
)

contact = ColumnToText(
    "contact",
    short_description="contact communication type for last contact of the current campaign",
    value_map=lambda x: f"{x}",
)

day = ColumnToText(
    "day",
    short_description="last contact day of the month",
    value_map=lambda x: f"{x} {'st' if x == 1 else 'nd' if x == 2 else 'rd' if x == 3 else 'th'} day of the month"
)

month = ColumnToText(
    "month",
    short_description="last contact month of year",
    value_map=lambda x: x
)

duration = ColumnToText(
    "duration",
    short_description="last contact duration, in seconds",
    value_map=lambda x: f"{x} s"
)

campaign = ColumnToText(
    "campaign",
    short_description="number of contacts performed during this campaign and for this client",
    value_map=lambda x: f"{x} contacts"
)

pdays = ColumnToText(
    "pdays",
    short_description="number of days that passed by after the client was last contacted from a previous campaign",
    value_map={
        x: x if x != -1 else "client not previously contacted" for x in set(data["pdays"])
    }
)

previous = ColumnToText(
    "previous",
    short_description="number of contacts performed before this campaign and for this client",
    value_map=lambda x: x
)

poutcome = ColumnToText(
    "poutcome",
    short_description="outcome of the previous marketing campaign",
    value_map=lambda x: x
)

y = ColumnToText(
    "y_binary",
    short_description="has the client subscribed a term deposit?",
    value_map={
        0: "no",
        1: "yes"
    }
)


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}

reentry_numeric_qa = DirectNumericQA(
    column='y_binary',
    text=(
        "Has the client subscribed a term deposit?"
    ),
)


reentry_qa = MultipleChoiceQA(
    column='y_binary',
    text="Has the client subscribed a term deposit?",
    choices=(
        Choice("Yes, they have", 1),
        Choice("No, they have not", 0),
    ),
)



all_outcomes = ["y_binary"]

reentry_task = TaskMetadata(
    name="y prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='y_binary',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

# we want to sample 5k
subsampling = 5000 / num_data

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

    RESULTS_DIR = "bank"
    bench.run(results_root_dir=RESULTS_DIR)

