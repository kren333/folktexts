#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pip install ucimlrepo
@author: jingjingtang
"""
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 

from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset

# categorical_vars = [
#     "buying",    # Buying price: vhigh, high, med, low
#     "maint",     # Maintenance cost: vhigh, high, med, low
#     "doors",     # Number of doors: 2, 3, 4, 5more
#     "persons",   # Capacity in terms of persons to carry: 2, 4, more
#     "lug_boot",  # Size of luggage boot: small, med, big
#     "safety",    # Estimated safety of the car: low, med, high
#     "class"      # Car acceptability: unacc, acc, good, vgood # round to 0 or 1 (acc, good, vgood are all considered as acc)
# ]

# continuous_vars = [] 



TASK_DESCRIPTION = """\
The following data corresponds to evaluations of cars based on various categorical features. \
Each entry represents a car described by its buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety rating. \
Please answer each question based on the information provided. \
The data is sufficient to determine the acceptability of each car according to predefined criteria.
"""



buying_dict = {
    "vhigh": "very high",
    "high": "high",
    "med": "medium",
    "low": "low"
}

maint_dict = {
    "vhigh": "very high",
    "high": "high",
    "med": "medium",
    "low": "low"
}

doors_dict = {
    "2": "two",
    "3": "three",
    "4": "four",
    "5more": "five or more"
}

persons_dict = {
    "2": "two people",
    "4": "four people",
    "more": "more than four people"
}

lug_boot_dict = {
    "small": "small",
    "med": "medium",
    "big": "large"
}

safety_dict = {
    "low": "low",
    "med": "medium",
    "high": "high"
}

# class_dict = {
#     "unacc": "unacceptable",
#     "acc": "acceptable",
#     "good": "good",
#     "vgood": "very good"
# }
class_numeric_dict = {
    0: "unacceptable",
    1: "acceptable"
}


buying_col = ColumnToText(
    "buying",
    short_description="buying price",
    value_map={k: f"The buying price is {v}" for k, v in buying_dict.items()}
)

maint_col = ColumnToText(
    "maint",
    short_description="price of the maintenance",
    value_map={k: f"The price of the maintenance is {v}" for k, v in maint_dict.items()}
)

doors_col = ColumnToText(
    "doors",
    short_description="number of doors",
    value_map={k: f"The car has {v} doors" for k, v in doors_dict.items()}
)

persons_col = ColumnToText(
    "persons",
    short_description="capacity in terms of persons to carry",
    value_map={k: f"The car holds {v}" for k, v in persons_dict.items()}
)

lug_boot_col = ColumnToText(
    "lug_boot",
    short_description="size of the luggage boot",
    value_map={k: f"The size of the luggage boot is {v}" for k, v in lug_boot_dict.items()}
)

safety_col = ColumnToText(
    "safety",
    short_description="estimated safety of the car",
    value_map={k: f"The safety rating is {v}" for k, v in safety_dict.items()}
)

# class_col = ColumnToText(
#     "class",
#     short_description="evaluation level",
#     value_map={k: f"The car is {v}" for k, v in class_dict.items()}
# )
class_numeric_col = ColumnToText(
    "class_numeric",
    short_description="evaluation level",
    value_map={k: f"The car is {v}" for k, v in class_numeric_dict.items()}
)



# car_eval_direct_qa = DirectNumericQA(
#     column="class",
#     text="How acceptable is this car?"
# )

# car_eval_mc_qa = MultipleChoiceQA(
#     column="class",
#     text="How acceptable is this car?",
#     choices=(
#         Choice("The car is considered to be unacceptable", "unacc"),
#         Choice("The car is considered to be acceptable", "acc"),
#         Choice("The car is considered to be good", "good"),
#         Choice("The car is considered to be very good", "vgood"),
#     ),
# )

car_eval_direct_qa = DirectNumericQA(
    column="class_numeric",
    text="How acceptable is this car?"
)

car_eval_mc_qa = MultipleChoiceQA(
    column="class_numeric",
    text="How acceptable is this car?",
    choices=(
        Choice("The car is considered to be unacceptable", 0),
        Choice("The car is considered to be acceptable", 1),
        # Choice("The car is considered to be good", 2),
        # Choice("The car is considered to be very good", 3),
    ),
)


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}



all_outcomes = ["class_numeric"]

reentry_task = TaskMetadata(
    name="car evaluation",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='class_numeric',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=car_eval_mc_qa,
    direct_numeric_qa=car_eval_direct_qa,
)


car_evaluation = fetch_ucirepo(id=19) 
data = car_evaluation.data.original
# The class columns is categorical, use a numeric one for the numeric QA
class_to_int = {
    "unacc": 0,
    "acc": 1,
    "good": 1,
    "vgood": 1
}
# Add numeric version to the DataFrame
data["class_numeric"] = data["class"].map(class_to_int)

# num_data = len(data)
# # we want to sample 10k
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
with open("secrets.txt", "r") as handle:
    secrets = json.load(handle)
os.environ["OPENAI_API_KEY"] = secrets["open_ai_key"]
    
for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "uci-car_evaluation"
    bench.run(results_root_dir=RESULTS_DIR)
