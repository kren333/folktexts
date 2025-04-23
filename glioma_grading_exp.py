#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pip install ucimlrepo

@author: jingjingtang
"""
import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset

# Define variable types
# categorical_vars = [
#     'Grade', 'Gender', 'Race',
#     'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC',
#     'FUBP1', 'PIK3CA', 'NF1', 'PIK3R1', 'PDGFRA',
#     'SMARCA4', 'RB1', 'PIK3CB', 'PIK3CD', 'PIK3CG',
#     'PIK3R2', 'PIK3R3', 'PIK3R5', 'PIK3R6'
# ]

# continuous_vars = ['Age_at_diagnosis']  


TASK_DESCRIPTION = """\
The following data corresponds to clinical and genetic features of patients diagnosed with glioma. \
This includes patient demographics (such as age, gender, and race) as well as the mutation status of various genes commonly associated with glioma. \
Each entry describes a unique patient. \
Please answer the following question based on the information provided. \
The data is sufficient to determine whether the patient has a low-grade glioma (LGG) or glioblastoma (GBM).
"""

  

# Target variable: Glioma Grade
grade_dict = {
    0: "low-grade glioma (LGG)",
    1: "glioblastoma (GBM)"
}

# Gender
gender_dict = {
    0: "male",
    1: "female"
}

# Race
race_dict = {
    0: "White",
    1: "Black or African American",
    2: "Asian",
    3: "American Indian or Alaska Native"
}

# Gene mutation status
mutation_dict = {
    0: "NOT_MUTATED",
    1: "MUTATED"
}

# Grade (target)
grade_col = ColumnToText(
    "Grade",
    short_description="glioma grade",
    value_map={k: f"The patient has a {v}" for k, v in grade_dict.items()}
)

# Gender
gender_col = ColumnToText(
    "Gender",
    short_description="gender",
    value_map={k: f"The patient is {v}" for k, v in gender_dict.items()}
)

# Age at diagnosis
age_col = ColumnToText(
    "Age_at_diagnosis",
    short_description="age at diagnosis",
    value_map=lambda x: f"The patient was diagnosed at {float(x)} years old"
)

# Race
race_col = ColumnToText(
    "Race",
    short_description="race",
    value_map={k: f"The patient's race is {v}" for k, v in race_dict.items()}
)


IDH1_col = ColumnToText(
    "IDH1",
    short_description="IDH1 mutation status",
    value_map={
        0: "IDH1 (isocitrate dehydrogenase (NADP(+)) 1) is NOT_MUTATED",
        1: "IDH1 (isocitrate dehydrogenase (NADP(+)) 1) is MUTATED"
    }
)

TP53_col = ColumnToText(
    "TP53",
    short_description="TP53 mutation status",
    value_map={
        0: "TP53 (tumor protein p53) is NOT_MUTATED",
        1: "TP53 (tumor protein p53) is MUTATED"
    }
)

ATRX_col = ColumnToText(
    "ATRX",
    short_description="ATRX mutation status",
    value_map={
        0: "ATRX (ATRX chromatin remodeler) is NOT_MUTATED",
        1: "ATRX (ATRX chromatin remodeler) is MUTATED"
    }
)

PTEN_col = ColumnToText(
    "PTEN",
    short_description="PTEN mutation status",
    value_map={
        0: "PTEN (phosphatase and tensin homolog) is NOT_MUTATED",
        1: "PTEN (phosphatase and tensin homolog) is MUTATED"
    }
)

EGFR_col = ColumnToText(
    "EGFR",
    short_description="EGFR mutation status",
    value_map={
        0: "EGFR (epidermal growth factor receptor) is NOT_MUTATED",
        1: "EGFR (epidermal growth factor receptor) is MUTATED"
    }
)

CIC_col = ColumnToText(
    "CIC",
    short_description="CIC mutation status",
    value_map={
        0: "CIC (capicua transcriptional repressor) is NOT_MUTATED",
        1: "CIC (capicua transcriptional repressor) is MUTATED"
    }
)

MUC16_col = ColumnToText(
    "MUC16",
    short_description="MUC16 mutation status",
    value_map={
        0: "MUC16 (mucin 16, cell surface associated) is NOT_MUTATED",
        1: "MUC16 (mucin 16, cell surface associated) is MUTATED"
    }
)

PIK3CA_col = ColumnToText(
    "PIK3CA",
    short_description="PIK3CA mutation status",
    value_map={
        0: "PIK3CA (phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha) is NOT_MUTATED",
        1: "PIK3CA (phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha) is MUTATED"
    }
)

NF1_col = ColumnToText(
    "NF1",
    short_description="NF1 mutation status",
    value_map={
        0: "NF1 (neurofibromin 1) is NOT_MUTATED",
        1: "NF1 (neurofibromin 1) is MUTATED"
    }
)

PIK3R1_col = ColumnToText(
    "PIK3R1",
    short_description="PIK3R1 mutation status",
    value_map={
        0: "PIK3R1 (phosphoinositide-3-kinase regulatory subunit 1) is NOT_MUTATED",
        1: "PIK3R1 (phosphoinositide-3-kinase regulatory subunit 1) is MUTATED"
    }
)

FUBP1_col = ColumnToText(
    "FUBP1",
    short_description="FUBP1 mutation status",
    value_map={
        0: "FUBP1 (far upstream element binding protein 1) is NOT_MUTATED",
        1: "FUBP1 (far upstream element binding protein 1) is MUTATED"
    }
)

RB1_col = ColumnToText(
    "RB1",
    short_description="RB1 mutation status",
    value_map={
        0: "RB1 (RB transcriptional corepressor 1) is NOT_MUTATED",
        1: "RB1 (RB transcriptional corepressor 1) is MUTATED"
    }
)

NOTCH1_col = ColumnToText(
    "NOTCH1",
    short_description="NOTCH1 mutation status",
    value_map={
        0: "NOTCH1 (notch receptor 1) is NOT_MUTATED",
        1: "NOTCH1 (notch receptor 1) is MUTATED"
    }
)

BCOR_col = ColumnToText(
    "BCOR",
    short_description="BCOR mutation status",
    value_map={
        0: "BCOR (BCL6 corepressor) is NOT_MUTATED",
        1: "BCOR (BCL6 corepressor) is MUTATED"
    }
)

CSMD3_col = ColumnToText(
    "CSMD3",
    short_description="CSMD3 mutation status",
    value_map={
        0: "CSMD3 (CUB and Sushi multiple domains 3) is NOT_MUTATED",
        1: "CSMD3 (CUB and Sushi multiple domains 3) is MUTATED"
    }
)

SMARCA4_col = ColumnToText(
    "SMARCA4",
    short_description="SMARCA4 mutation status",
    value_map={
        0: "SMARCA4 (SWI/SNF related, matrix associated, actin dependent regulator of chromatin, subfamily a, member 4) is NOT_MUTATED",
        1: "SMARCA4 (SWI/SNF related, matrix associated, actin dependent regulator of chromatin, subfamily a, member 4) is MUTATED"
    }
)

GRIN2A_col = ColumnToText(
    "GRIN2A",
    short_description="GRIN2A mutation status",
    value_map={
        0: "GRIN2A (glutamate ionotropic receptor NMDA type subunit 2A) is NOT_MUTATED",
        1: "GRIN2A (glutamate ionotropic receptor NMDA type subunit 2A) is MUTATED"
    }
)

IDH2_col = ColumnToText(
    "IDH2",
    short_description="IDH2 mutation status",
    value_map={
        0: "IDH2 (isocitrate dehydrogenase (NADP(+)) 2) is NOT_MUTATED",
        1: "IDH2 (isocitrate dehydrogenase (NADP(+)) 2) is MUTATED"
    }
)

FAT4_col = ColumnToText(
    "FAT4",
    short_description="FAT4 mutation status",
    value_map={
        0: "FAT4 (FAT atypical cadherin 4) is NOT_MUTATED",
        1: "FAT4 (FAT atypical cadherin 4) is MUTATED"
    }
)

PDGFRA_col = ColumnToText(
    "PDGFRA",
    short_description="PDGFRA mutation status",
    value_map={
        0: "PDGFRA (platelet-derived growth factor receptor alpha) is NOT_MUTATED",
        1: "PDGFRA (platelet-derived growth factor receptor alpha) is MUTATED"
    }
)

glioma_grade_direct_qa = DirectNumericQA(
    column="Grade",
    text="What is the grade of this glioma?"
)

glioma_grade_mc_qa = MultipleChoiceQA(
    column="Grade",
    text="What is the grade of this glioma?",
    choices=(
        Choice("The patient has a low-grade glioma (LGG)", 0),
        Choice("The patient has a glioblastoma (GBM)", 1),
    ),
)


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


all_outcomes = ["Grade"]

reentry_task = TaskMetadata(
    name="glioma grading",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='Grade',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=glioma_grade_mc_qa,
    direct_numeric_qa=glioma_grade_direct_qa,
)

# fetch dataset 
glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759) 
data = glioma_grading_clinical_and_mutation_features.data.original
# data.to_csv("uci-glioma_grading.csv", index=False)
# Define reverse mapping
race_reverse_map = {
    "white": 0,
    "black or african american": 1,
    "asian": 2,
    "american indian or alaska native": 3
}

# Apply to your DataFrame
data["Race"] = data["Race"].str.lower().map(race_reverse_map)

# num_data = len(data)
# # we want to sample 10k
# subsampling = 50000 / num_data

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
    llm_clf = WebAPILLMClassifier(
        model_name=model_name, 
        task=task,
        custom_prompt_prefix=TASK_DESCRIPTION)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "uci-glioma_grading"
    bench.run(results_root_dir=RESULTS_DIR)

