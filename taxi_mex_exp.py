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

# os.chdir('/users/bryanwilder/Dropbox/llm_preds')


TASK_DESCRIPTION = """\
The following data corresponds to taxi routes, Uber and Cabify from Mexico City. \
The task is to predict whether the total ride duration time exceeds 30 minutes, based on location and temporal features.\
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each ride.
"""
pickup_datetime_col = ColumnToText(
    "pickup_datetime",
    short_description="pickup date and time",
    value_map=lambda x: f"pickup occurred on {x.strftime('%A, %B %d, %Y at %I:%M %p')}"
)

pickup_longitude_col = ColumnToText(
    "pickup_longitude",
    short_description="pickup longitude",
    value_map=lambda x: f"pickup longitude at {x:.5f}"
)

pickup_latitude_col = ColumnToText(
    "pickup_latitude",
    short_description="pickup latitude",
    value_map=lambda x: f"pickup latitude at {x:.5f}"
)

dropoff_longitude_col = ColumnToText(
    "dropoff_longitude",
    short_description="dropoff longitude",
    value_map=lambda x: f"dropoff longitude at {x:.5f}"
)

dropoff_latitude_col = ColumnToText(
    "dropoff_latitude",
    short_description="dropoff latitude",
    value_map=lambda x: f"dropoff latitude at {x:.5f}"
)

dist_meters_col = ColumnToText(
    "dist_meters",
    short_description="trip distance in meters",
    value_map=lambda x: f"trip covered {x} meters"
)

wait_sec_col = ColumnToText(
    "wait_sec",
    short_description="waiting time in seconds",
    value_map=lambda x: f"waited {x} seconds before pickup"
)

month_col = ColumnToText(
    "month",
    short_description="month of pickup",
    value_map=lambda x: f"pickup occurred in month {x}"
)

week_col = ColumnToText(
    "week",
    short_description="week number of pickup",
    value_map=lambda x: f"pickup occurred in week {x}"
)

weekday_col = ColumnToText(
    "weekday",
    short_description="day of the week",
    value_map=lambda x: f"pickup occurred on weekday {x}"
)

hour_col = ColumnToText(
    "hour",
    short_description="hour of pickup",
    value_map=lambda x: f"pickup occurred at hour {x}"
)

minute_oftheday_col = ColumnToText(
    "minute_oftheday",
    short_description="minute of the day",
    value_map=lambda x: f"pickup occurred at minute {x} of the day"
)

trip_duration_col = ColumnToText(
    "trip_duration",
    short_description="trip duration exceed 30 minutes (1800 seconds)",
    value_map={0: 'No', 1: 'Yes'}
)

reentry_numeric_qa = DirectNumericQA(
    column='trip_duration',
    text=(
        "Does the total ride duration time exceed 30 minutes (1800 seconds)? Answer 1 for yes, 0 for no."
    ),
)

reentry_qa = MultipleChoiceQA(
    column='trip_duration',
    text="Does the total ride duration time exceed 30 minutes (1800 seconds)?",
    choices=(
        Choice("Yes, the total ride duration time exceeds 30 minutes (1800 seconds)", 1),
        Choice("No, the total ride duration time does not exceed 30 minutes (1800 seconds)", 0),
    ),
)


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}

all_outcomes = ["trip_duration"]

reentry_task = TaskMetadata(
    name="trip duration prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='trip_duration',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

df_taxi = pd.read_csv("data/taxi_mex.csv")
df_taxi = df_taxi[(df_taxi.trip_duration < 5900)]
df_taxi = df_taxi[(df_taxi.pickup_longitude > -110)]
df_taxi = df_taxi[(df_taxi.pickup_latitude < 50)]
df_taxi.drop(['store_and_fwd_flag'], axis=1, inplace=True)
df_taxi.drop(['vendor_id'], axis=1, inplace=True)
df_taxi['pickup_datetime'] = pd.to_datetime(df_taxi.pickup_datetime)
df_taxi.drop(['dropoff_datetime'], axis=1, inplace=True) 
df_taxi['month'] = df_taxi.pickup_datetime.dt.month
df_taxi['week'] = df_taxi.pickup_datetime.dt.isocalendar().week
df_taxi['weekday'] = df_taxi.pickup_datetime.dt.weekday
df_taxi['hour'] = df_taxi.pickup_datetime.dt.hour
df_taxi['minute'] = df_taxi.pickup_datetime.dt.minute
df_taxi['minute_oftheday'] = df_taxi['hour'] * 60 + df_taxi['minute']
df_taxi.drop(['minute'], axis=1, inplace=True)
data = df_taxi
data['trip_duration'] = data['trip_duration'].apply(lambda x: 1 if x > 1800 else 0)
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
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "taxi_mex"
    bench.run(results_root_dir=RESULTS_DIR)



# llm_clf = WebAPILLMClassifier(model_name=model_name, task=shelter_task)
# bench = Benchmark(llm_clf=llm_clf, dataset=shelter_dataset)
# RESULTS_DIR = "res_shelter"
# bench.run(results_root_dir=RESULTS_DIR)
