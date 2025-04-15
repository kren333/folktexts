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

# # Define variable types
# categorical_vars = [
#     'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#     'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
#     'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#     'stalk-surface-below-ring', 'stalk-color-above-ring',
#     'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
#     'ring-type', 'spore-print-color', 'population', 'habitat'
# ]

# continuous_vars = [
#     # Add simulated continuous variable(s) here if any exist
#     # e.g., 'cap_diameter' if you engineered one
# ]


TASK_DESCRIPTION = """\
The following data describes physical and environmental characteristics of different mushroom species. \
Each entry includes attributes such as cap shape, odor, gill size, spore print color, and more. \
The goal is to determine whether a mushroom is edible or poisonous based on these features. \
Please answer the following question using the information provided.
"""


mushroom_feature_dicts = {
    "poisonous": {
        0: "edible",
        1: "poisonous"
    },
    "cap-shape": {
        "b": "bell",
        "c": "conical",
        "x": "convex",
        "f": "flat",
        "k": "knobbed",
        "s": "sunken"
    },
    "cap-surface": {
        "f": "fibrous",
        "g": "grooves",
        "y": "scaly",
        "s": "smooth"
    },
    "cap-color": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "r": "green",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "bruises": {
        "t": "bruises",
        "f": "no bruises"
    },
    "odor": {
        "a": "almond",
        "l": "anise",
        "c": "creosote",
        "y": "fishy",
        "f": "foul",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy"
    },
    "gill-attachment": {
        "a": "attached",
        "d": "descending",
        "f": "free",
        "n": "notched"
    },
    "gill-spacing": {
        "c": "close",
        "w": "crowded",
        "d": "distant"
    },
    "gill-size": {
        "b": "broad",
        "n": "narrow"
    },
    "gill-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "g": "gray",
        "r": "green",
        "o": "orange",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "stalk-shape": {
        "e": "enlarging",
        "t": "tapering"
    },
    "stalk-root": {
        "b": "bulbous",
        "c": "club",
        "u": "cup",
        "e": "equal",
        "z": "rhizomorphs",
        "r": "rooted",
        "m": "missing"
    },
    "stalk-surface-above-ring": {
        "f": "fibrous",
        "y": "scaly",
        "k": "silky",
        "s": "smooth"
    },
    "stalk-surface-below-ring": {
        "f": "fibrous",
        "y": "scaly",
        "k": "silky",
        "s": "smooth"
    },
    "stalk-color-above-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "stalk-color-below-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "veil-type": {
        "p": "partial",
        "u": "universal"
    },
    "veil-color": {
        "n": "brown",
        "o": "orange",
        "w": "white",
        "y": "yellow"
    },
    "ring-number": {
        "n": "none",
        "o": "one",
        "t": "two"
    },
    "ring-type": {
        "c": "cobwebby",
        "e": "evanescent",
        "f": "flaring",
        "l": "large",
        "n": "none",
        "p": "pendant",
        "s": "sheathing",
        "z": "zone"
    },
    "spore-print-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "r": "green",
        "o": "orange",
        "u": "purple",
        "w": "white",
        "y": "yellow"
    },
    "population": {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several",
        "y": "solitary"
    },
    "habitat": {
        "g": "grasses",
        "l": "leaves",
        "m": "meadows",
        "p": "paths",
        "u": "urban",
        "w": "waste",
        "d": "woods"
    }
}


poisonous_col = ColumnToText(
    "poisonous",
    short_description="poisonous",
    value_map={k: f"The mushroom is {v}" for k, v in mushroom_feature_dicts["poisonous"].items()}
)

cap_shape_col = ColumnToText(
    "cap-shape",
    short_description="cap shape",
    value_map={k: f"The mushroom's cap shape is {v}" for k, v in mushroom_feature_dicts["cap-shape"].items()}
)

cap_surface_col = ColumnToText(
    "cap-surface",
    short_description="cap surface",
    value_map={k: f"The mushroom's cap surface is {v}" for k, v in mushroom_feature_dicts["cap-surface"].items()}
)

cap_color_col = ColumnToText(
    "cap-color",
    short_description="cap color",
    value_map={k: f"The mushroom's cap color is {v}" for k, v in mushroom_feature_dicts["cap-color"].items()}
)

bruises_col = ColumnToText(
    "bruises",
    short_description="bruises",
    value_map={k: f"The mushroom's bruises is {v}" for k, v in mushroom_feature_dicts["bruises"].items()}
)

odor_col = ColumnToText(
    "odor",
    short_description="odor",
    value_map={k: f"The mushroom's odor is {v}" for k, v in mushroom_feature_dicts["odor"].items()}
)

gill_attachment_col = ColumnToText(
    "gill-attachment",
    short_description="gill attachment",
    value_map={k: f"The mushroom's gill attachment is {v}" for k, v in mushroom_feature_dicts["gill-attachment"].items()}
)

gill_spacing_col = ColumnToText(
    "gill-spacing",
    short_description="gill spacing",
    value_map={k: f"The mushroom's gill spacing is {v}" for k, v in mushroom_feature_dicts["gill-spacing"].items()}
)

gill_size_col = ColumnToText(
    "gill-size",
    short_description="gill size",
    value_map={k: f"The mushroom's gill size is {v}" for k, v in mushroom_feature_dicts["gill-size"].items()}
)

gill_color_col = ColumnToText(
    "gill-color",
    short_description="gill color",
    value_map={k: f"The mushroom's gill color is {v}" for k, v in mushroom_feature_dicts["gill-color"].items()}
)

stalk_shape_col = ColumnToText(
    "stalk-shape",
    short_description="stalk shape",
    value_map={k: f"The mushroom's stalk shape is {v}" for k, v in mushroom_feature_dicts["stalk-shape"].items()}
)

stalk_root_col = ColumnToText(
    "stalk-root",
    short_description="stalk root",
    value_map={k: f"The mushroom's stalk root is {v}" for k, v in mushroom_feature_dicts["stalk-root"].items()}
)

stalk_surface_above_ring_col = ColumnToText(
    "stalk-surface-above-ring",
    short_description="stalk surface above ring",
    value_map={k: f"The mushroom's stalk surface above ring is {v}" for k, v in mushroom_feature_dicts["stalk-surface-above-ring"].items()}
)

stalk_surface_below_ring_col = ColumnToText(
    "stalk-surface-below-ring",
    short_description="stalk surface below ring",
    value_map={k: f"The mushroom's stalk surface below ring is {v}" for k, v in mushroom_feature_dicts["stalk-surface-below-ring"].items()}
)

stalk_color_above_ring_col = ColumnToText(
    "stalk-color-above-ring",
    short_description="stalk color above ring",
    value_map={k: f"The mushroom's stalk color above ring is {v}" for k, v in mushroom_feature_dicts["stalk-color-above-ring"].items()}
)

stalk_color_below_ring_col = ColumnToText(
    "stalk-color-below-ring",
    short_description="stalk color below ring",
    value_map={k: f"The mushroom's stalk color below ring is {v}" for k, v in mushroom_feature_dicts["stalk-color-below-ring"].items()}
)

veil_type_col = ColumnToText(
    "veil-type",
    short_description="veil type",
    value_map={k: f"The mushroom's veil type is {v}" for k, v in mushroom_feature_dicts["veil-type"].items()}
)

veil_color_col = ColumnToText(
    "veil-color",
    short_description="veil color",
    value_map={k: f"The mushroom's veil color is {v}" for k, v in mushroom_feature_dicts["veil-color"].items()}
)

ring_number_col = ColumnToText(
    "ring-number",
    short_description="ring number",
    value_map={k: f"The mushroom's ring number is {v}" for k, v in mushroom_feature_dicts["ring-number"].items()}
)

ring_type_col = ColumnToText(
    "ring-type",
    short_description="ring type",
    value_map={k: f"The mushroom's ring type is {v}" for k, v in mushroom_feature_dicts["ring-type"].items()}
)

spore_print_color_col = ColumnToText(
    "spore-print-color",
    short_description="spore print color",
    value_map={k: f"The mushroom's spore print color is {v}" for k, v in mushroom_feature_dicts["spore-print-color"].items()}
)

population_col = ColumnToText(
    "population",
    short_description="population",
    value_map={k: f"The mushroom's population is {v}" for k, v in mushroom_feature_dicts["population"].items()}
)

habitat_col = ColumnToText(
    "habitat",
    short_description="habitat",
    value_map={k: f"The mushroom's habitat is {v}" for k, v in mushroom_feature_dicts["habitat"].items()}
)


reentry_numeric_qa = DirectNumericQA(
    column="poisonous",
    text="Is this mushroom poisonous or edible?"
)

reentry_qa = MultipleChoiceQA(
    column="poisonous",
    text="Is this mushroom poisonous or edible?",
    choices=(
        Choice("The mushroom is edible", 0),
        Choice("The mushroom is poisonous", 1),
    ),
)




columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


# all_outcomes = ['JAIL', 'FOUR_ER', 'EMERG_SHLTR', 'MHIP']
all_outcomes = ["poisonous"]

reentry_task = TaskMetadata(
    name="mushroom classiciation",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='poisonous',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)


# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
data = mushroom.data.original
class_to_int = {
    "e": 0,
    "p": 1
}
# Add numeric version to the DataFrame
data["poisonous"] = data["poisonous"].map(class_to_int)
data["stalk-root"].fillna("m")
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

    RESULTS_DIR = "uci-mushroom"
    bench.run(results_root_dir=RESULTS_DIR)

