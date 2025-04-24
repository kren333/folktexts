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
The following data corresponds to nursery school application evaluations. \
Each record includes social, financial, and personal information about a child's family situation. \
We are aiming to classify each application into two categories: as priority or not. \
Please answer each question based on the information provided. \
The data provided is sufficient to make an informed decision for each application.
"""

parents_col = ColumnToText(
    "parents",
    short_description="Quality of parents",
    value_map={
        "usual": "Usual parental care",
        "pretentious": "Pretentious parental care",
        "great_pret": "Highly pretentious parental care"
    }
)

has_nurs_col = ColumnToText(
    "has_nurs",
    short_description="Nursery service availability",
    value_map={
        "proper": "Proper nursery care",
        "less_proper": "Less proper nursery care",
        "improper": "Improper nursery care",
        "critical": "Critical nursery need",
        "very_crit": "Very critical nursery need"
    }
)

form_col = ColumnToText(
    "form",
    short_description="Completion of formal application",
    value_map={
        "complete": "Fully complete application",
        "completed": "Recently completed application",
        "incomplete": "Incomplete application",
        "foster": "Foster care situation"
    }
)

children_col = ColumnToText(
    "children",
    short_description="Number of children",
    value_map={
        "1": "1 child",
        "2": "2 children",
        "3": "3 children",
        "more": "More than 3 children"
    }
)

housing_col = ColumnToText(
    "housing",
    short_description="Housing situation",
    value_map={
        "convenient": "Convenient housing",
        "less_conv": "Less convenient housing",
        "critical": "Critical housing condition"
    }
)

finance_col = ColumnToText(
    "finance",
    short_description="Financial situation",
    value_map={
        "convenient": "Financially convenient",
        "inconv": "Financially inconvenient"
    }
)

social_col = ColumnToText(
    "social",
    short_description="Social conditions",
    value_map={
        "nonprob": "No social problems",
        "slightly_prob": "Slight social problems",
        "problematic": "Problematic social conditions"
    }
)

health_col = ColumnToText(
    "health",
    short_description="Health status",
    value_map={
        "recommended": "Recommended health status",
        "priority": "Priority health case",
        "not_recom": "Not recommended health status"
    }
)

acceptable_col = ColumnToText(
    "recommendation_priority",
    short_description="admission recommendation priority",
    value_map={
        0: "not priority",
        1: "priority"
    }
)

reentry_numeric_qa = DirectNumericQA(
    column='recommendation_priority',
    text=(
        "Is this application classified as priority?"
    ),
)

reentry_qa = MultipleChoiceQA(
    column='recommendation_priority',
    text="Is this application classified as priority?",
    choices=(
        Choice("Yes, it's classified as priority.", 1),
        Choice("No, it's not classified as priority.", 0),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["recommendation_priority"]
discretize_cols = []

reentry_task = TaskMetadata(
    name="Nursery placement priority or not",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='recommendation_priority',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

data = pd.read_csv("../data/nursery.csv")
data['recommendation_priority'] = data['recommendation_priority'].map({'not_recom': 0, 'recommend': 0, 'very_recom': 0, 'priority': 1, 'spec_prior': 1})
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

    RESULTS_DIR = "nursery"
    bench.run(results_root_dir=RESULTS_DIR)
