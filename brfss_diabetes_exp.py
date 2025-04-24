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
def brfss_shared_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['_RFHYPE5'].isin([1, 2])]
    df = df[df['_CHOLCHK'].isin([1, 2])]
    df = df[df['BPHIGH4'].isin([1, 2, 3, 4])]
    df = df[df['DIABETE3'].isin([1, 3])]
    df = df[df['_RFCHOL'].isin([1, 2])]
    df = df[df['_AGEG5YR'].isin([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])]
    return df

# os.chdir('/users/bryanwilder/Dropbox/llm_preds')

TASK_DESCRIPTION = """\
The following data corresponds to Behavioral Risk Factor Surveillance System (BRFSS). \
BRFSS is a large-scale telephone survey conducted by the Centers of Disease Control and Prevention. \
BRFSS collects data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. \
BRFSS collects data in all 50 states as well as the District of Columbia and three U.S. territories. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

state_dict={
    1: 'Alabama',
    2: 'Alaska',
    4: 'Arizona',
    5: 'Arkansas',
    6: 'California',
    8: 'Colorado',
    9: 'Connecticut',
    10: 'Delaware',
    11: 'District of Columbia',
    12: 'Florida',
    13: 'Georgia',
    15: 'Hawaii',
    16: 'Idaho',
    17: 'Illinois ',
    18: 'Indiana',
    19: 'Iowa',
    20: 'Kansas',
    21: 'Kentucky',
    22: 'Louisiana ',
    23: 'Maine',
    24: 'Maryland',
    25: 'Massachusetts',
    26: 'Michigan',
    27: 'Minnesota',
    28: 'Mississippi',
    29: 'Missouri',
    30: 'Montana',
    31: 'Nebraska',
    32: 'Nevada',
    33: 'New Hampshire',
    34: 'New Jersey',
    35: 'New Mexico',
    36: 'New York',
    37: 'North Carolina',
    38: 'North Dakota',
    39: 'Ohio',
    40: 'Oklahoma',
    41: 'Oregon',
    42: 'Pennsylvania',
    44: 'Rhode Island',
    45: 'South Carolina',
    46: 'South Dakota',
    47: 'Tennessee',
    48: 'Texas',
    49: 'Utah',
    50: 'Vermont',
    51: 'Virginia',
    53: 'Washington',
    54: 'West Virginia',
    55: 'Wisconsin',
    56: 'Wyoming',
    66: 'Guam',
    72: 'Puerto Rico'
}

state_col = ColumnToText(
    "_STATE",
    short_description="State FIPS Code",
    value_map=state_dict
)

genhealth_col = ColumnToText(
    "_RFHLTH",
    short_description="In general your health is",
    value_map={
        1: 'Good or Better Health',
        2: 'Fair or Poor Health',
        9: "Don't know/Not Sure Or Refused/Missing"
    }
)

persdoc_col = ColumnToText(
    "PERSDOC2",
    short_description="Have one person you think of as your personal doctor or health care provider",
    value_map={
        1: "Yes, only one.", 2: "More than one", 3: "No", 7: "Don't know/not sure", 9: "Refused",
    }
)

medcost_col = ColumnToText(
    "MEDCOST",
    short_description="Was there a time in the past 12 months when needing to see a doctor but could not because of cost?",
    value_map={
        1: "Yes", 2: "No", 7: "Don't know/not sure", 9: "Refused",
    }
)

checkup_col = ColumnToText(
    "CHECKUP1",
    short_description="Time since last visit to the doctor for a checkup",
    value_map={
        1: 'Within past year (anytime < 12 months ago)',
        2: 'Within past 2 years (1 year but < 2 years ago)',
        3: 'Within past 5 years (2 years but < 5 years ago)',
        4: '5 or more years ago',
        7: "Don't know/Not sure", 8: 'Never', 9: 'Refused'
    }
)

cholchk_col = ColumnToText(
    "_CHOLCHK",
    short_description="Time since last blood cholesterol check",
    value_map={
        1: 'Had cholesterol checked in past 5 years',
        2: 'Did not have cholesterol checked in past 5 years',
        3: 'Have never had cholesterol checked',
        9: "Don't know/Not Sure Or Refused/Missing"
    }
)

race_col = ColumnToText(
    "_PRACE1",
    short_description="Preferred race category",
    value_map={
        1: 'White',
        2: 'Black or African American',
        3: 'American Indian or Alaskan Native',
        4: 'Asian', 
        5: 'Native Hawaiian or other Pacific Islander',
        6: 'Other race',
        7: 'No preferred race',
        8: 'Multiracial but preferred race not answered',
        77: "Don't know/Not sure", 99: 'refused', 
    },
)

sex_col = ColumnToText(
    "SEX",
    short_description="Sex of respondent",
    value_map={1: "Male", 2: "Female"},
)

fruit_col = ColumnToText(
    "_FRTLT1",
    short_description="Consume Fruit 1 or more times per day",
    value_map={
        1: 'Consumed fruit one or more times per day',
        2: 'Consumed fruit less than one  time per day',
        9: "Don't know, refused or missing values"
    }
)

veg_col = ColumnToText(
    "_VEGLT1",
    short_description="Consume vegetables 1 or more times per day",
    value_map={
        1: 'Consumed vegetables one or more times per day',
        2: 'Consumed vegetables less  than one time per day',
        9: "Don't know, refused or missing values",
    }
)

binge_col = ColumnToText(
    "_RFBING5",
    short_description="Respondent is binge drinker (males having five or more drinks on one occasion, females having four or more drinks on one occasion)",
    value_map={1: "No", 2: "Yes", 9: "Don't know/Refused/Missing"}
)

smoke100_col = ColumnToText(
    "SMOKE100",
    short_description="Have smoked at least 100 cigarettes in the entire life?",
    value_map={
        1: 'Every day', 2: 'Some days', 3: 'Not at all',
        7: "Don't Know/Not Sure",
        9: 'Refused'
    },
)

smoke_col = ColumnToText(
    "SMOKDAY2",
    short_description="Now smoke cigarettes every day, some days, or not at all?",
    value_map={1: "Every day", 2: "Some days", 3: "Not at all", 7: "Don't Know/Not Sure", 9: "Refused"}
)

bmi_col = ColumnToText(
    "_BMI5CAT",
    short_description="Body Mass Index (BMI) category",
    value_map={
        1: 'Underweight (BMI < 1850)',
        2: 'Normal Weight (1850 <= BMI < 2500)',
        3: 'Overweight (2500 <= BMI < 3000)',
        4: 'Obese (3000 <= BMI < 9999)'
    }
)

activity_col = ColumnToText(
    "_TOTINDA",
    short_description="Adults who reported doing physical activity or exercise during the past 30 days other than their regular job.",
    value_map={
        1: 'Had physical activity or exercise in last 30 days',
        2: 'No physical activity or exercise in last 30 days',
        9: "Dont know/Refused/Missing"
    }
)

income_col = ColumnToText(
    "INCOME2",
    short_description="Annual household income from all sources",
    value_map={
        1: 'Less than $10,000',
        2: 'Less than $15,000 ($10,000 to less than $15,000)',
        3: 'Less than $20,000 ($15,000 to less than $20,000)',
        4: 'Less than $25,000 ($20,000 to less than $25,000)',
        5: 'Less than $35,000 ($25,000 to less than $35,000)',
        6: 'Less than $50,000 ($35,000 to less than $50,000)',
        7: 'Less than $75, 000 ($50,000 to less than $75,000)',
        8: '$75,000 or more (BRFSS 2015-2019) or less than $100,000 ($75,000 to < $100,000) (BRFSS 2021)',
        9: 'Less than $150,000 ($100,000 to < $150,000)',
        10: 'Less than $200,000 ($150,000 to < $200,000)',
        11: '$200,000  or more',
        77: "Don't know/Not sure", 99: 'Refused',
    }
)

marital_col = ColumnToText(
    "MARITAL",
    short_description="Marital status",
    value_map={
        1: 'Married', 2: 'Divorced',
        3: 'Widowed', 4: 'Separated', 5: 'Never married',
        6: 'A member of an unmarried couple', 9: 'Refused'
    }
)

edu_col = ColumnToText(
    "EDUCA",
    short_description="Highest grade or year of school completed",
    value_map={
        1: 'Never attended school or only kindergarten',
        2: 'Grades 1 through 8 (Elementary)',
        3: 'Grades 9 through 11 (Some high school)',
        4: 'Grade 12 or GED (High school graduate)',
        5: 'College 1 year to 3 years (Some college or technical school)',
        6: 'College 4 years or more (College graduate)', 9: 'Refused'
    }
)

healthcov_col = ColumnToText(
    "_HCVU651",
    short_description="Current health care coverage",
    value_map={
        1: 'Have health care coverage',
        2: 'Do not have health care coverage',
        9: "Don't know/Not Sure, Refused or Missing"
    }
)

age_col = ColumnToText(
    "_AGEG5YR",
    short_description="Age group",
    value_map={
        1: 'Age 18 to 24', 2: 'Age 25 to 29', 3: ' Age 30 to 34',
        4: 'Age 35 to 39',
        5: 'Age 40 to 44', 6: 'Age 45 to 49', 7: 'Age 50 to 54',
        8: 'Age 55 to 59', 9: 'Age 60 to 64',
        10: 'Age 65 to 69', 11: 'Age 70 to 74',
        12: 'Age 75 to 79', 13: 'Age 80 or older',
        14: "Don't know/Refused/Missing"
    }
)

skinca_col = ColumnToText(
    "CHCSCNCR",
    short_description="Have skin cancer or ever told you have skin cancer",
    value_map={
        1: 'Yes', 2: 'No', 7: "Don't know/Not Sure", 9: 'Refused'
    }
)

otherca_col = ColumnToText(
    "CHCOCNCR",
    short_description="Have any other types of cancer or ever told you have any other types of cancer",
    value_map={
        1: 'Yes', 2: 'No', 7: "Don't know/Not Sure", 9: 'Refused'
    }
)

employ_col = ColumnToText(
    "EMPLOY1",
    short_description="Current employment status",
    value_map={
        1: 'Employed for wages', 
        2: 'Self-employed',
        3: 'Out of work for 1 year or more',
        4: 'Out of work for less than 1 year', 
        5: 'A homemaker',
        6: 'A student',
        7: 'Retired', 
        8: 'Unable to work', 
        9: 'Refused'
    }
)


asthma_col = ColumnToText(
    "_CASTHM1",
    short_description="Told having asthma",
    value_map={
        1: 'No', 2: 'Yes', 9: "Don't know/Not Sure/Refused/Missing"
    }
)

stroke_col = ColumnToText(
    "CVDSTRK3",
    short_description="Ever had a stroke, or been told had a stroke",
    value_map={
        1: 'Yes', 2: 'No', 7: "Don't know/Not Sure", 9: 'Refused'
    }
)

hbpgeneral_col = ColumnToText(
    "BPHIGH4",
    short_description="Hypertension Awareness: Ever told having high blood pressure",
    value_map={
        1: 'Yes', 2: 'Yes, but female told only during pregnancy', 
        3: 'No', 4: 'Told borderline high or pre-hypertensive', 
        7: "Don't know/not sure", 9: "Refused"
    }
)

hbp_col = ColumnToText(
    "_RFHYPE5",
    short_description="Told having high blood pressure",
    value_map={
        1: 'No', 2: 'Yes', 9: "Don't know/Not Sure/Refused/Missing"
    }
)

hbc_col = ColumnToText(
    "_RFCHOL",
    short_description="Told having high cholesterol",
    value_map={
        1: 'No', 2: 'Yes', 9: "Don't know/Not Sure/Refused/Missing"
    }
)

chdmi_col = ColumnToText(
    "_MICHD",
    short_description="Reports of coronary heart disease (CHD) or myocardial infarction (MI)",
    value_map={
        1: 'Reported having myocardial infarction or coronary heart disease',
        2: 'Did not report having myocardial infarction or coronary heart disease',
    }
)

diabetes_col = ColumnToText(
    "DIABETE3",
    short_description="(Ever told) you have diabetes",
    value_map={
        1: 'Yes',
        0: 'No'
    }
)

reentry_numeric_qa = DirectNumericQA(
    column='DIABETE3',
    text=(
        "Does this patient have diabetes?"
    ),
)

reentry_qa = MultipleChoiceQA(
    column='DIABETE3',
    text="Does this patient have diabetes?",
    choices=(
        Choice("Yes, they have been told having diabetes", 1),
        Choice("No, they have not been told having diabetes", 0),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["DIABETE3"]
discretize_cols = []

reentry_task = TaskMetadata(
    name="diabetes prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='DIABETE3',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

data = pd.read_csv("data/brfss.csv")
data = brfss_shared_preprocessing(data)
data = data[data["DIABETE3"].isin([1, 3])]
data['DIABETE3'] = data['DIABETE3'].map({1: 1, 3: 0})
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

    RESULTS_DIR = "brfss_diabetes"
    bench.run(results_root_dir=RESULTS_DIR)
