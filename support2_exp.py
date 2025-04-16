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

# os.chdir('/users/bryanwilder/Dropbox/llm_preds')


TASK_DESCRIPTION = """\
The following data corresponds 9105 individual critically ill patients across 5 United States medical centers, accessioned throughout 1989-1991 and 1992-1994. \
Each row concerns hospitalized patient records who met the inclusion and exclusion criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

"""COLUMNS"""
def num_to_text(unit=""):
    unit = f" {unit}" if unit else ""
    return lambda x: f"{x}{unit}" if pd.notna(x) else "missing"
age_col      = ColumnToText("age",      "age in years",                       value_map=num_to_text("yr"))
slos_col     = ColumnToText("slos",     "days from study entry to discharge", value_map=num_to_text("d"))
dtime_col    = ColumnToText("d.time",   "days of follow-up",                  value_map=num_to_text("d"))
numco_col    = ColumnToText("num.co",   "number of comorbidities",            value_map=num_to_text())
scoma_col    = ColumnToText("scoma",    "coma score day 3",                   value_map=num_to_text())
charges_col  = ColumnToText("charges",  "hospital charges",                   value_map=num_to_text("USD"))
totcst_col   = ColumnToText("totcst",   "total RCC cost",                     value_map=num_to_text("USD"))
totmcst_col  = ColumnToText("totmcst",  "total micro cost",                   value_map=num_to_text("USD"))
avtisst_col  = ColumnToText("avtisst",  "average TISS score days 3-25",       value_map=num_to_text())
sps_col      = ColumnToText("sps",      "SUPPORT physiology score day 3",     value_map=num_to_text())
aps_col      = ColumnToText("aps",      "APACHE III physiology score day 3",  value_map=num_to_text())
surv2m_col   = ColumnToText("surv2m",   "model 2-month survival estimate",    value_map=num_to_text())
surv6m_col   = ColumnToText("surv6m",   "model 6-month survival estimate",    value_map=num_to_text())
hday_col     = ColumnToText("hday",     "hospital day at study entry",        value_map=num_to_text())
prg2m_col    = ColumnToText("prg2m",    "physician 2-month survival estimate",value_map=num_to_text())
prg6m_col    = ColumnToText("prg6m",    "physician 6-month survival estimate",value_map=num_to_text())
dnrday_col   = ColumnToText("dnrday",   "day of DNR order",                   value_map=num_to_text())
meanbp_col   = ColumnToText("meanbp",   "mean arterial blood pressure",       value_map=num_to_text("mmHg"))
wblc_col     = ColumnToText("wblc",     "white blood cells (thousands)",      value_map=num_to_text("k"))
hrt_col      = ColumnToText("hrt",      "heart rate",                         value_map=num_to_text("bpm"))
resp_col     = ColumnToText("resp",     "respiration rate",                   value_map=num_to_text("breaths/min"))
temp_col     = ColumnToText("temp",     "body temperature",                   value_map=num_to_text("C"))
pafi_col     = ColumnToText("pafi",     "PaO2/FiO2 ratio",                    value_map=num_to_text())
alb_col      = ColumnToText("alb",      "serum albumin",                      value_map=num_to_text("g/dL"))
bili_col     = ColumnToText("bili",     "bilirubin",                          value_map=num_to_text("mg/dL"))
crea_col     = ColumnToText("crea",     "creatinine",                         value_map=num_to_text("mg/dL"))
sod_col      = ColumnToText("sod",      "serum sodium",                       value_map=num_to_text("mEq/L"))
ph_col       = ColumnToText("ph",       "arterial blood pH",                  value_map=num_to_text())
glucose_col  = ColumnToText("glucose",  "glucose",                            value_map=num_to_text("mg/dL"))
bun_col      = ColumnToText("bun",      "blood urea nitrogen",                value_map=num_to_text("mg/dL"))
urine_col    = ColumnToText("urine",    "urine output",                       value_map=num_to_text("mL"))
adls_col     = ColumnToText("adls",     "ADL index (surrogate)",              value_map=num_to_text())
adlsc_col    = ColumnToText("adlsc",    "imputed ADL calibrated",             value_map=num_to_text())

diabetes_col = ColumnToText("diabetes", "diabetes comorbidity",
                            value_map={1: "Yes", 0: "No", pd.NA: "unknown"})
dementia_col = ColumnToText("dementia", "dementia comorbidity",
                            value_map={1: "Yes", 0: "No", pd.NA: "unknown"})
hospdead_col = ColumnToText("hospdead", "death in hospital",
                            value_map={1: "Yes", 0: "No", pd.NA: "unknown"})
# death_col    = ColumnToText("death",    "death before 31‑Dec‑1994",
#                             value_map={1: "Yes", 0: "No", pd.NA: "unknown"})

sex_col     = ColumnToText("sex",     "sex",                value_map=lambda x: x if pd.notna(x) else "unknown")
dzgroup_col = ColumnToText("dzgroup", "disease subcategory",value_map=lambda x: x if pd.notna(x) else "unknown")
dzclass_col = ColumnToText("dzclass", "disease category",   value_map=lambda x: x if pd.notna(x) else "unknown")
edu_col     = ColumnToText("edu",     "years of education", value_map=num_to_text("yr"))
income_col  = ColumnToText("income",  "income bracket",     value_map=lambda x: x if pd.notna(x) else "unknown")
race_col    = ColumnToText("race",    "race",               value_map=lambda x: x if pd.notna(x) else "unknown")
ca_col      = ColumnToText("ca",      "cancer status",      value_map=lambda x: x if pd.notna(x) else "unknown")
dnr_col     = ColumnToText("dnr",     "DNR order status",   value_map=lambda x: x if pd.notna(x) else "unknown")
adlp_col    = ColumnToText("adlp",    "ADL index (patient)",value_map=num_to_text())
sfdm2_col   = ColumnToText("sfdm2",   "functional disability", value_map=lambda x: x if pd.notna(x) else "missing")


"""QUESTIONS"""


reentry_numeric_qa = DirectNumericQA(
    column="hospdead",
    text="Did the patient die in the hospital? Answer 1 for yes, 0 for no."
)


reentry_qa = MultipleChoiceQA(
    column="hospdead",
    text="Did the patient die in the hospital?",
    choices=(
        Choice("Yes, the patient died in hospital", 1),
        Choice("No, the patient survived to discharge", 0),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}

all_outcomes = ["hospdead"]

reentry_task = TaskMetadata(
    name="Hospital death prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='hospdead',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)


data = pd.read_csv("data/support2.csv")
data = data[data['hospdead'].isin([0, 1])]
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

    RESULTS_DIR = "support2"
    bench.run(results_root_dir=RESULTS_DIR)



# llm_clf = WebAPILLMClassifier(model_name=model_name, task=shelter_task)
# bench = Benchmark(llm_clf=llm_clf, dataset=shelter_dataset)
# RESULTS_DIR = "res_shelter"
# bench.run(results_root_dir=RESULTS_DIR)
