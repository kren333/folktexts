#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pip install ucimlrepo

@author: jingjingtang
"""
import os
import json
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 

import folktexts
from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset


TASK_DESCRIPTION = """\
The following data contains morphological measurements of individual rice grains. \
Each entry includes features such as area, perimeter, axis lengths, eccentricity, convex area, and extent, which describe the shape and geometry of the grain. \
The goal is to predict the variety of the rice grain—either Cammeo or Osmancik—based on these features. \
Please answer the following question using the information provided. \
The data is sufficient to distinguish between the two rice varieties.
"""

area_col = ColumnToText(
    "Area",
    short_description="number of pixels within the boundaries of the rice grain",
    value_map=lambda x: f"The number of pixels within the boundaries of the rice grain is {round(float(x), 2)}"
)

perimeter_col = ColumnToText(
    "Perimeter",
    short_description="circumference calculated by the distance between pixels around the grain's boundary",
    value_map=lambda x: f"The circumference calculated by the distance between pixels around the grain's boundary is {round(float(x), 2)}"
)

major_axis_length_col = ColumnToText(
    "Major_Axis_Length",
    short_description="length of the longest line that can be drawn on the rice grain (main axis)",
    value_map=lambda x: f"The length of the longest line that can be drawn on the rice grain (main axis) is {round(float(x), 2)}"
)

minor_axis_length_col = ColumnToText(
    "Minor_Axis_Length",
    short_description="length of the shortest line that can be drawn on the rice grain (minor axis)",
    value_map=lambda x: f"The length of the shortest line that can be drawn on the rice grain (minor axis) is {round(float(x), 2)}"
)

eccentricity_col = ColumnToText(
    "Eccentricity",
    short_description="roundness of the ellipse with the same moments as the rice grain",
    value_map=lambda x: f"The roundness of the ellipse with the same moments as the rice grain is {round(float(x), 2)}"
)

convex_area_col = ColumnToText(
    "Convex_Area",
    short_description="pixel count of the smallest convex shell enclosing the rice grain",
    value_map=lambda x: f"The pixel count of the smallest convex shell enclosing the rice grain is {round(float(x), 2)}"
)

extent_col = ColumnToText(
    "Extent",
    short_description="ratio of the region formed by the rice grain to its bounding box",
    value_map=lambda x: f"The ratio of the region formed by the rice grain to its bounding box is {round(float(x), 2)}"
)

class_col = ColumnToText(
    "Class",
    short_description="rice variety",
    value_map={
        0: "The rice variety is Cammeo",
        1: "The rice variety is Osmancik"
    }
)


reentry_numeric_qa = DirectNumericQA(
    column="Class",
    text="What is the variety of this rice grain?"
)

reentry_qa = MultipleChoiceQA(
    column="Class",
    text="What is the variety of this rice grain?",
    choices=(
        Choice("The rice variety is Cammeo", 0),
        Choice("The rice variety is Osmancik", 1),
    ),
)



columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}

all_outcomes = ["Class"]

reentry_task = TaskMetadata(
    name="rice variety classiciation",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='Class',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)


# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
data = rice_cammeo_and_osmancik.data.original
class_to_int = {
    "Cammeo": 0,
    "Osmancik": 1
}
# Add numeric version to the DataFrame
data["Class"] = data["Class"].map(class_to_int)
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

with open("secrets.txt", "r") as handle:
    secrets = json.load(handle)
os.environ["OPENAI_API_KEY"] = secrets["open_ai_key"]
    
for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "uci-rice"
    bench.run(results_root_dir=RESULTS_DIR)

