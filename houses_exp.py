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
The following dataset contains housing information from California census blocks from the 1990 Census. \
In this sample a block group on average includes 1425.5 individuals living in a geographically compact area. \
Each record includes demographic, geographic, and housing-related attributes. \
The task is to classify whether the median house value in a neighborhood in California, 1990 greater than 200,000, \
Use the provided attributes to make an informed classification.
"""

median_house_value_col = ColumnToText(
    "median_house_value",
    short_description="Median house value in the neighborhood in California, 1990",
    value_map={
        0: "not greater than 200,000",
        1: "greater than 200,000"
    }
)

median_income_col = ColumnToText(
    "median_income",
    short_description="Median household income in the neighborhood in California, 1990",
    value_map=lambda x: x
)

housing_median_age_col = ColumnToText(
    "housing_median_age",
    short_description="Median age of the houses in the neighborhood in California, 1990",
    value_map=lambda x: x
)

total_rooms_col = ColumnToText(
    "total_rooms",
    short_description="Total number of rooms in the block",
    value_map=lambda x: x
)

total_bedrooms_col = ColumnToText(
    "total_bedrooms",
    short_description="Total number of bedrooms in the block",
    value_map=lambda x: x
)

population_col = ColumnToText(
    "population",
    short_description="Total population in the block",
    value_map=lambda x: x
)

households_col = ColumnToText(
    "households",
    short_description="Number of households in the block",
    value_map=lambda x: x
)

latitude_col = ColumnToText(
    "latitude",
    short_description="Geographical latitude of the block",
    value_map=lambda x: x
)

latitude_col = ColumnToText(
    "longitude",
    short_description="Geographical longitude of the block",
    value_map=lambda x: x
)

reentry_numeric_qa = DirectNumericQA(
    column='median_house_value',
    text=(
        "Is the median housing value greater than 200,000?"
    ),
)

reentry_qa = MultipleChoiceQA(
    column='median_house_value',
    text="Is the housing value greater than 200,000?",
    choices=(
        Choice("Yes, the housing value is greater than 200,000.", 1),
        Choice("No, the housing value is not greater than 200,000.", 0),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["median_house_value"]
discretize_cols = ["median_income", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "latitude", "longitude"]

reentry_task = TaskMetadata(
    name="median house value classification",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='median_house_value',
    cols_to_text=columns_map,
    sensitive_attribute=None, 
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

data = pd.read_csv("../data/houses.csv")
data['median_house_value'] = data['median_house_value'].apply(lambda x: 1 if x > 200000 else 0)
num_data = len(data)
# we want to sample 10k
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
os.environ["OPENAI_API_KEY"] = json.loads("secrets.txt")["open_ai_key"]

for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task, custom_prompt_prefix=TASK_DESCRIPTION)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "houses"
    bench.run(results_root_dir=RESULTS_DIR)
