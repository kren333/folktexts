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

TASK_DESCRIPTION = """\
The following data corresponds to car evaluation records derived from a hierarchical decision-making model. \
Each record describes characteristics related to the car’s price and technical features, including comfort and safety. \
The goal is to classify each car as either acceptable or not acceptable for use. \
Please answer each question based on the features provided. \
The data is comprehensive enough to support a well-informed classification.
"""

# buying,maint,doors,persons,lug_boot,safety,acceptable
buying_col = ColumnToText(
    "buying",
    short_description="Initial buying price of the car",
    value_map={
        "vhigh": "Very high buying price",
        "high": "High buying price",
        "med": "Medium buying price",
        "low": "Low buying price"
    }
)

maint_col = ColumnToText(
    "maint",
    short_description="Ongoing maintenance cost",
    value_map={
        "vhigh": "Very high maintenance cost",
        "high": "High maintenance cost",
        "med": "Medium maintenance cost",
        "low": "Low maintenance cost"
    }
)

doors_col = ColumnToText(
    "doors",
    short_description="Number of doors",
    value_map={
        "2": "2-door car",
        "3": "3-door car",
        "4": "4-door car",
        "5more": "Car with 5 or more doors"
    }
)

persons_col = ColumnToText(
    "persons",
    short_description="Passenger capacity",
    value_map={
        "2": "Seats 2 people",
        "4": "Seats 4 people",
        "more": "Seats more than 4 people"
    }
)

lug_boot_col = ColumnToText(
    "lug_boot",
    short_description="Size of the luggage boot",
    value_map={
        "small": "Small luggage capacity",
        "med": "Medium luggage capacity",
        "big": "Large luggage capacity"
    }
)

safety_col = ColumnToText(
    "safety",
    short_description="Estimated safety rating",
    value_map={
        "low": "Low safety rating",
        "med": "Moderate safety rating",
        "high": "High safety rating"
    }
)

acceptable_col = ColumnToText(
    "acceptable",
    short_description="Car acceptability",
    value_map={
        0: "not acceptable",
        1: "acceptable"
    }
)


reentry_numeric_qa = DirectNumericQA(
    column='acceptable',
    text=(
        "Is the car acceptable?"
    ),
)

reentry_qa = MultipleChoiceQA(
    column='acceptable',
    text="Is the car acceptable?",
    choices=(
        Choice("Yes, the car is acceptable.", 1),
        Choice("No, the car is not acceptable.", 0),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["acceptable"]
discretize_cols = []

reentry_task = TaskMetadata(
    name="car acceptability classification",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='acceptable',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

data = pd.read_csv("../data/car.csv")
data['acceptable'] = data['acceptable'].map({'unacc': 0, 'acc': 1, 'good': 1, 'vgood': 1})
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

    RESULTS_DIR = "car"
    bench.run(results_root_dir=RESULTS_DIR)
