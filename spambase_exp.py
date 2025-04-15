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
# continuous_vars = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
#                    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
#                    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
#                    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
#                    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
#                    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
#                    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
#                    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
#                    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
#                    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
#                    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
#                    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
#                    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$',
#                    'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
#                    'capital_run_length_total']
# categorical_vars = ['spam']  # The target variable


TASK_DESCRIPTION = """\
The following data corresponds to email messages represented by statistical features extracted from their text content. \
The data includes word and character frequency measures, as well as information about capital letter usage, for emails collected and labeled as spam or not spam. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each email.
""" 
word_freq_make_col = ColumnToText(
    "word_freq_make",
    short_description="word frequency of 'make'",
    value_map=lambda x: f"The word 'make' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_address_col = ColumnToText(
    "word_freq_address",
    short_description="word frequency of 'address'",
    value_map=lambda x: f"The word 'address' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_all_col = ColumnToText(
    "word_freq_all",
    short_description="word frequency of 'all'",
    value_map=lambda x: f"The word 'all' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_3d_col = ColumnToText(
    "word_freq_3d",
    short_description="word frequency of '3d'",
    value_map=lambda x: f"The word '3d' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_our_col = ColumnToText(
    "word_freq_our",
    short_description="word frequency of 'our'",
    value_map=lambda x: f"The word 'our' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_over_col = ColumnToText(
    "word_freq_over",
    short_description="word frequency of 'over'",
    value_map=lambda x: f"The word 'over' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_remove_col = ColumnToText(
    "word_freq_remove",
    short_description="word frequency of 'remove'",
    value_map=lambda x: f"The word 'remove' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_internet_col = ColumnToText(
    "word_freq_internet",
    short_description="word frequency of 'internet'",
    value_map=lambda x: f"The word 'internet' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_order_col = ColumnToText(
    "word_freq_order",
    short_description="word frequency of 'order'",
    value_map=lambda x: f"The word 'order' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_mail_col = ColumnToText(
    "word_freq_mail",
    short_description="word frequency of 'mail'",
    value_map=lambda x: f"The word 'mail' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_receive_col = ColumnToText(
    "word_freq_receive",
    short_description="word frequency of 'receive'",
    value_map=lambda x: f"The word 'receive' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_will_col = ColumnToText(
    "word_freq_will",
    short_description="word frequency of 'will'",
    value_map=lambda x: f"The word 'will' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_people_col = ColumnToText(
    "word_freq_people",
    short_description="word frequency of 'people'",
    value_map=lambda x: f"The word 'people' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_report_col = ColumnToText(
    "word_freq_report",
    short_description="word frequency of 'report'",
    value_map=lambda x: f"The word 'report' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_addresses_col = ColumnToText(
    "word_freq_addresses",
    short_description="word frequency of 'addresses'",
    value_map=lambda x: f"The word 'addresses' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_free_col = ColumnToText(
    "word_freq_free",
    short_description="word frequency of 'free'",
    value_map=lambda x: f"The word 'free' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_business_col = ColumnToText(
    "word_freq_business",
    short_description="word frequency of 'business'",
    value_map=lambda x: f"The word 'business' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_email_col = ColumnToText(
    "word_freq_email",
    short_description="word frequency of 'email'",
    value_map=lambda x: f"The word 'email' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_you_col = ColumnToText(
    "word_freq_you",
    short_description="word frequency of 'you'",
    value_map=lambda x: f"The word 'you' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_credit_col = ColumnToText(
    "word_freq_credit",
    short_description="word frequency of 'credit'",
    value_map=lambda x: f"The word 'credit' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_your_col = ColumnToText(
    "word_freq_your",
    short_description="word frequency of 'your'",
    value_map=lambda x: f"The word 'your' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_font_col = ColumnToText(
    "word_freq_font",
    short_description="word frequency of 'font'",
    value_map=lambda x: f"The word 'font' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_000_col = ColumnToText(
    "word_freq_000",
    short_description="word frequency of '000'",
    value_map=lambda x: f"The word '000' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_money_col = ColumnToText(
    "word_freq_money",
    short_description="word frequency of 'money'",
    value_map=lambda x: f"The word 'money' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_hp_col = ColumnToText(
    "word_freq_hp",
    short_description="word frequency of 'hp'",
    value_map=lambda x: f"The word 'hp' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_hpl_col = ColumnToText(
    "word_freq_hpl",
    short_description="word frequency of 'hpl'",
    value_map=lambda x: f"The word 'hpl' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_george_col = ColumnToText(
    "word_freq_george",
    short_description="word frequency of 'george'",
    value_map=lambda x: f"The word 'george' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_650_col = ColumnToText(
    "word_freq_650",
    short_description="word frequency of '650'",
    value_map=lambda x: f"The word '650' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_lab_col = ColumnToText(
    "word_freq_lab",
    short_description="word frequency of 'lab'",
    value_map=lambda x: f"The word 'lab' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_labs_col = ColumnToText(
    "word_freq_labs",
    short_description="word frequency of 'labs'",
    value_map=lambda x: f"The word 'labs' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_telnet_col = ColumnToText(
    "word_freq_telnet",
    short_description="word frequency of 'telnet'",
    value_map=lambda x: f"The word 'telnet' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_857_col = ColumnToText(
    "word_freq_857",
    short_description="word frequency of '857'",
    value_map=lambda x: f"The word '857' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_data_col = ColumnToText(
    "word_freq_data",
    short_description="word frequency of 'data'",
    value_map=lambda x: f"The word 'data' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_415_col = ColumnToText(
    "word_freq_415",
    short_description="word frequency of '415'",
    value_map=lambda x: f"The word '415' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_85_col = ColumnToText(
    "word_freq_85",
    short_description="word frequency of '85'",
    value_map=lambda x: f"The word '85' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_technology_col = ColumnToText(
    "word_freq_technology",
    short_description="word frequency of 'technology'",
    value_map=lambda x: f"The word 'technology' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_1999_col = ColumnToText(
    "word_freq_1999",
    short_description="word frequency of '1999'",
    value_map=lambda x: f"The word '1999' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_parts_col = ColumnToText(
    "word_freq_parts",
    short_description="word frequency of 'parts'",
    value_map=lambda x: f"The word 'parts' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_pm_col = ColumnToText(
    "word_freq_pm",
    short_description="word frequency of 'pm'",
    value_map=lambda x: f"The word 'pm' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_direct_col = ColumnToText(
    "word_freq_direct",
    short_description="word frequency of 'direct'",
    value_map=lambda x: f"The word 'direct' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_cs_col = ColumnToText(
    "word_freq_cs",
    short_description="word frequency of 'cs'",
    value_map=lambda x: f"The word 'cs' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_meeting_col = ColumnToText(
    "word_freq_meeting",
    short_description="word frequency of 'meeting'",
    value_map=lambda x: f"The word 'meeting' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_original_col = ColumnToText(
    "word_freq_original",
    short_description="word frequency of 'original'",
    value_map=lambda x: f"The word 'original' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_project_col = ColumnToText(
    "word_freq_project",
    short_description="word frequency of 'project'",
    value_map=lambda x: f"The word 'project' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_re_col = ColumnToText(
    "word_freq_re",
    short_description="word frequency of 're'",
    value_map=lambda x: f"The word 're' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_edu_col = ColumnToText(
    "word_freq_edu",
    short_description="word frequency of 'edu'",
    value_map=lambda x: f"The word 'edu' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_table_col = ColumnToText(
    "word_freq_table",
    short_description="word frequency of 'table'",
    value_map=lambda x: f"The word 'table' appears in {round(float(x), 2)}% of all words in the email"
)

word_freq_conference_col = ColumnToText(
    "word_freq_conference",
    short_description="word frequency of 'conference'",
    value_map=lambda x: f"The word 'conference' appears in {round(float(x), 2)}% of all words in the email"
)

char_freq_semicolon_col = ColumnToText(
    "char_freq_;",
    short_description="frequency of character ';'",
    value_map=lambda x: f"The character ';' appears in {round(float(x), 2)}% of all characters in the email"
)

char_freq_paren_col = ColumnToText(
    "char_freq_(",
    short_description="frequency of character '('",
    value_map=lambda x: f"The character '(' appears in {round(float(x), 2)}% of all characters in the email"
)

char_freq_bracket_col = ColumnToText(
    "char_freq_[",
    short_description="frequency of character '['",
    value_map=lambda x: f"The character '[' appears in {round(float(x), 2)}% of all characters in the email"
)

char_freq_exclam_col = ColumnToText(
    "char_freq_!",
    short_description="frequency of character '!'",
    value_map=lambda x: f"The character '!' appears in {round(float(x), 2)}% of all characters in the email"
)

char_freq_dollar_col = ColumnToText(
    "char_freq_$",
    short_description="frequency of character '$'",
    value_map=lambda x: f"The character '$' appears in {round(float(x), 2)}% of all characters in the email"
)

char_freq_hash_col = ColumnToText(
    "char_freq_#",
    short_description="frequency of character '#'",
    value_map=lambda x: f"The character '#' appears in {round(float(x), 2)}% of all characters in the email"
)


capital_run_length_average_col = ColumnToText(
    "capital_run_length_average",
    short_description="average capital run length",
    value_map=lambda x: f"The average capital run length is {round(float(x), 1)}"
)

capital_run_length_longest_col = ColumnToText(
    "capital_run_length_longest",
    short_description="longest capital run length",
    value_map=lambda x: f"The longest uninterrupted capital letter run is {int(x)} characters"
)

capital_run_length_total_col = ColumnToText(
    "capital_run_length_total",
    short_description="total capital letters",
    value_map=lambda x: f"The total number of capital letters is {int(x)}"
)




spam_col = ColumnToText(
    "Class",
    short_description="spam label",
    value_map={
        0: "The email is not spam",
        1: "The email is spam"
    }
)

spambase_direct_qa = DirectNumericQA(
    column="spam",
    text="Is this email spam or not?"
)

spambase_mc_qa = MultipleChoiceQA(
    column="spam",
    text="Is this email spam or not?",
    choices=(
        Choice("The email is not spam", 0),
        Choice("The email is spam", 1),
    ),
)

columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}

all_outcomes = ["Class"]

reentry_task = TaskMetadata(
    name="email spam classiciation",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='Class',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=spambase_mc_qa,
    direct_numeric_qa=spambase_direct_qa,
)


# fetch dataset 
spambase = fetch_ucirepo(id=94) 
data = spambase.data.original
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

    RESULTS_DIR = "uci-spambase"
    bench.run(results_root_dir=RESULTS_DIR)
