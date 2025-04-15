from classes import zero_shot_experiment
from tqdm import tqdm
from openai import OpenAI
import jsonlines
import json
import re
import pdb

delivery_map = {
    21: 'at a public hospital',
    22: 'at a private hospital',
    23: 'at a health center or clinic',
    24: 'at a maternity home or birthing center',
    25: 'at a health facility',
    26: 'at a health facility',
    11: 'at home',
    12: 'at the home of family member or friend',
    13: 'at the home of traditional birth attendant',
    31: 'en route to facility',
    32: 'at their workplace',
    33: 'outdoors',
    34: 'at an unspecified, nonconventional location',
    35: 'at an unspecified, nonconventional location',
    36: 'at an unspecified, nonconventional location',
    96: 'at an unspecified location',
    98: 'at an unspecified location',
    41: 'in a vehicle',
    42: 'in a temporary shelter',
    43: 'in a public space',
    44: 'at an unspecified, nonconventional location'
}

sex_map = {
    1: 'male',
    2: 'female'
}

urban_map = {
    1: 'urban',
    2: 'rural'
}

education_map = {
    0: 'has no formal education',
    1: 'attended up to primary school',
    2: 'attended up to secondary school',
    3: 'attended up to post-secondary school'
}

country_map = {
    'BO': 'Bolivia',
    'CO': 'Colombia',
    'DR': 'Dominican Republic',
    'GY': 'Guyana',
    'HT': 'Haiti',
    'HN': 'Honduras',
    'PE': 'Peru'
}

wealth_map = {
    1: 'first',
    2: 'second',
    3: 'third',
    4: 'fourth',
    5: 'fifth'
}

facility_map = {
    0: "outside of",
    1: "in"
}

def df_to_text(df):

    strings = []

    # for each row in the dataframe, convert to a string; return as list
    for idx in tqdm([i for i in range(len(df))]):
        bad_flag = False

        row = df.iloc[idx]

        care_str = ""
        for care_col in "nurse", "doctor", "aux_nurse":
            if row[care_col]:
                care_str += care_col
                care_str += " and "
        care_str = care_str[:-5]
        if not care_str:
            care_str = "no doctor or nurse"

        s = f"The mother is {row['mother_age']} years old from {row['Country']}, {row['mother_height'] / 10} cm tall, and weighs {row['mother_weight'] / 10} kg, has {'not' if not row['past_terminate'] else ''} had previously terminated pregnancy(s)"
        s += f", lives in a(n) {urban_map[int(row['urban'])]} environment and {education_map[int(row['mother_education'])]}; during pregnancy, was {'not' if not row['heathcare_visit'] else ''} visited by a family planning worker in the past year, had {row['antenatal_visits']} antenatal visits in total, "
        s += f"with care given by {care_str} during pregnancy"
        s += f"; the {sex_map[int(row['sex'])]} child was child number {row['birth_num']} born by the mother."
        s = s.replace("\n", "")
        strings.append(s)

    return strings


def process_batch(file_response):

    # client = OpenAI(api_key=api_key)
    # file_response = client.files.content(batch)
    # with jsonlines.Reader(file_response.text.splitlines()) as reader:
    #     file_response = [obj for obj in reader]
    
    num_weights = len(file_response)
    weights = []

    for response in tqdm(file_response): # TODO put weights in order; by the key
        try:
            _, res = response.split("Probability:")
            weights.append(float(res))
            # left_idx = gpt_output.find('[')
            # right_idx = gpt_output.find(']')
            # weights.append(float(gpt_output[left_idx+1:right_idx]))
        except:
            pdb.set_trace()
    assert(len(weights) == num_weights)
    return weights

if __name__ == "__main__":
    # train_csv_path = "/data/user_data/kevinren/kr_dfs/BO.csv"
    # src = "BO"
    column_ordering = [
            "antenatal_visits", 
            "birth_num", 
            "sex", 
            "past_terminate", 
            "heathcare_visit", 
            "mother_education", 
            "doctor", 
            "nurse", 
            "aux_nurse", 
            "mother_weight", 
            "mother_height", 
            "mother_age", 
            "urban",
            "wealth_index",
            "facility",
            "Country"
        ]
    discrete_columns = [
        'place_delivery',
        'sex',
        'past_terminate',
        'healthcare_visit',
        'mother_education',
        'doctor',
        'nurse',
        'aux_nurse',
        'c_section',
        'urban',
        'wealth_index',
        'facility',
        'Country'
    ]
    outcome = "facility"
    serialization_fn = df_to_text
    additional_info_fn = None
    tag = "risk_score_ipums_try_1"
    experiment_name = "ipums"
    context =  """You are an expert on international maternal health outcomes. You are assisting with the training of a machine learning model used to predict 
    whether a birth occurred in or not in a medical facility.""".replace("\n", "")
    # prompt = lambda src: """Predict whether the child was born in a medical facility or not (1 for yes, 0 for no), given that the mother is a resident of the country {src} in the early 2000s (2005 - 2009). Next,
    #                         assess/rate your confidence in this prediction with a number from 1 to 10, where 1 means you are completely unsure of your prediction and 10 means you are completely sure. For
    #                         the confidence score ONLY, return your answer between brackets [].""".replace("\n", "")
    THE_QUESTION="Predict whether the following child was born in a medical facility or not, given the following description and that the pregnancy took place between 2005 and 2009."
    prompt = lambda src: f"""Provide your best guess and the probability that it is correct (0.0 to 1.0) for
            the following question. Give ONLY the guess and probability, no other words or
            explanation. For example:\n\nGuess: <most likely guess, as short as possible; not
            a complete sentence, just the guess!>\n Probability: <the probability between 0.0
            and 1.0 that your guess is correct, without any extra commentary whatsoever; just
            the probability!>\n\nThe question is: ${THE_QUESTION}"""
    
    process_batch_fn = process_batch
    with open("../secrets.txt", "r") as handle:
        api_key = json.load("secrets.txt")["open_ai_key"]

    train_csv_path = f"data/ipums.csv"

    experiment = zero_shot_experiment(train_csv_path, column_ordering, discrete_columns, experiment_name, outcome, context, serialization_fn, additional_info_fn, tag, prompt, process_batch_fn, api_key)
    experiment.get_weights()