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
The task is to predict whether a given flight will be delayed, given the information of the scheduled departure.\
The data provided is enough to reach an approximate answer for each person.
"""

time = ColumnToText(
    "Time",
    short_description="number of minutes departure is past 12:00 am",
    value_map=lambda x: x
)

dayofweek = ColumnToText(
    "DayOfWeek",
    short_description="day of week",
    value_map=lambda x: x
)

airportto = ColumnToText(
    "AirportTo",
    short_description="airport of arrival",
    value_map=lambda x: x
)

airportfrom = ColumnToText(
    "AirportFrom",
    short_description="airport of departure",
    value_map=lambda x: x
)

flight = ColumnToText(
    "Flight",
    short_description="flight number",
    value_map=lambda x: x
)

airline = ColumnToText(
    "Airline",
    short_description="airline name",
    value_map=lambda x: x
)

delay = ColumnToText(
    "Delay",
    short_description="whether flight was delayed",
    value_map={
        1: "yes",
        0: "no"
    },
)


reentry_numeric_qa = DirectNumericQA(
    column='Delay',
    text=(
        "Did this flight get delayed?"
    ),
)


reentry_qa = MultipleChoiceQA(
    column='Delay',
    text="Did this flight get delayed?",
    choices=(
        Choice("Yes, it got delayed", 1),
        Choice("No, it did not get delayed", 0),
    ),
)


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["Delay"]

reentry_task = TaskMetadata(
    name="delay prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='Delay',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

data = pd.read_csv("data/airline.csv")
num_data = len(data)
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


if __name__ == "__main__":
    model_name = "openai/gpt-4o-mini"
    import os
    import json
    with open("secrets.json", "r") as handle:
        os.environ["OPENAI_API_KEY"] = json.load(handle)["open_ai_key"]
    # print("asdfasdf")
    # import pdb; pdb.set_trace()
        
    for taskname in all_tasks:
        task, dataset = all_tasks[taskname]
        llm_clf = WebAPILLMClassifier(model_name=model_name, task=task, custom_prompt_prefix=TASK_DESCRIPTION)
        llm_clf.set_inference_kwargs(batch_size=500)
        bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

        RESULTS_DIR = "airline"
        bench.run(results_root_dir=RESULTS_DIR)
