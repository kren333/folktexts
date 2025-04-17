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
import litellm
litellm._turn_on_debug()

TASK_DESCRIPTION = """\
The following data corresponds to rice grain's classification. \
Rice grain's images were taken for the two species, processed and feature inferences were made. Morphological features were obtained for each grain of rice. \
When  looking  at  the  general  characteristics  of  Osmancik species, they have a wide, long, glassy and dull appearance.  
When looking at the general characteristics of the Cammeo species, they have wide and long, glassy and dull in appearance. \     
The data provided is sufficient to make an informed decision for each application.
"""

# os.chdir('/users/bryanwilder/Dropbox/llm_preds')
area_col = ColumnToText(
    "Area",
    short_description="the number of pixels within the boundaries of the rice grain",
    value_map=lambda x: f"{x} pixels"
)

perimeter_col = ColumnToText(
    "Perimeter",
    short_description="total length around the rice grain boundary",
    value_map=lambda x: f"{x:.2f} units"
)

major_axis_col = ColumnToText(
    "Major_Axis_Length",
    short_description="length of the longest axis of the rice grain",
    value_map=lambda x: f"{x:.2f} units"
)

minor_axis_col = ColumnToText(
    "Minor_Axis_Length",
    short_description="length of the shortest axis of the rice grain",
    value_map=lambda x: f"{x:.2f} units"
)

eccentricity_col = ColumnToText(
    "Eccentricity",
    short_description="ratio describing the elongation of the rice grain",
    value_map=lambda x: f"{x:.4f}"
)

convex_area_col = ColumnToText(
    "Convex_Area",
    short_description="number of pixels in the convex hull of the rice grain",
    value_map=lambda x: f"{x} pixels"
)

extent_col = ColumnToText(
    "Extent",
    short_description="ratio of area to bounding box area of the rice grain",
    value_map=lambda x: f"{x:.4f}"
)


Cammeo_col = ColumnToText(
    "is_Cammeo",
    short_description="label: is Cammeo (1) or Osmancik (0)",
    value_map={
        0: "Osmancik",
        1: "Cammeo"
    }
)


reentry_numeric_qa = DirectNumericQA(
    column='is_Cammeo',
    text=(
        "Is the rice grain Cammeo rather than Osmancik?"
    ),
)


reentry_qa = MultipleChoiceQA(
    column='is_Cammeo',
    text="Is the rice grain Cammeo rather than Osmancik?",
    choices=(
        Choice("Yes, the rice grain is Cammeo", 1),
        Choice("No, the rice grain is Osmancik", 0),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["is_Cammeo"]

reentry_task = TaskMetadata(
    name="rice grain prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='is_Cammeo',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)


data = pd.read_csv("../data/Rice_Cammeo_Osmancik.csv")
import numpy as np
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
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "Rice_Cammeo_Osmancik"
    bench.run(results_root_dir=RESULTS_DIR)



# llm_clf = WebAPILLMClassifier(model_name=model_name, task=shelter_task)
# bench = Benchmark(llm_clf=llm_clf, dataset=shelter_dataset)
# RESULTS_DIR = "res_shelter"
# bench.run(results_root_dir=RESULTS_DIR)
