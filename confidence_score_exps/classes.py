import numpy as np
import os
import pandas as pd
import json
from tqdm import tqdm
import pdb
from openai import OpenAI
from tqdm import tqdm

def np_to_csv(arr, indices, column_ordering, discrete_columns):
    arr = arr[indices,:]

    df = pd.DataFrame(
        data=arr,
        columns=column_ordering
    )

    for col in column_ordering:
        if col in discrete_columns:
            df[col] = df[col].astype("category")
        else:
            print(col)
            df[col] = df[col].astype("float64")

    return df

class zero_shot_experiment:
    def __init__(self, train_csv_path, column_ordering, discrete_columns, experiment_name, outcome, context, serialization_fn, additional_info_fn, tag, prompt_fn, process_batch_fn, api_key):
        self.train_csv = pd.read_csv(train_csv_path)[column_ordering]
        self.column_ordering = column_ordering
        self.discrete_columns = discrete_columns
        self.outcome = outcome
        self.serialization_fn = serialization_fn # use for turning row of csv into text
        if additional_info_fn:
            self.additional_info = additional_info_fn(self.train_csv) # assumes that the additional_info_fn takes in the 2 csvs and does analysis on them
        else:
            self.additional_info = None        
        self.tag = tag # use for making the gpt call
        self.prompt_fn = prompt_fn
        self.process_batch = process_batch_fn
        self.api_key = api_key
        self.context = context
        self.experiment_name = experiment_name
    
    def get_weights(self):
        client = OpenAI(api_key=self.api_key)

        # turn the df to text and save

        np.random.seed(5)
        train_indices = np.random.choice(len(self.train_csv), size=min(1000, len(self.train_csv)), replace=False)

        train = np_to_csv(self.train_csv.to_numpy(), train_indices, self.column_ordering, self.discrete_columns)

        strs = self.serialization_fn(train)

        # pass each string through gpt; ask to reweight

        context = self.context

        prompt = self.prompt_fn("")
        if self.additional_info: prompt += self.additional_info

        # create json file

        # jsons = []
        results = []

        for i, s in tqdm(enumerate(strs)):

            # if i == 5: break

            full_prompt = f"{s} {prompt}"
            messages = [
                                {"role": "system", "content": context},
                                {"role": "user", "content": full_prompt}
                            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages
            )
            results.append(response.choices[0].message.content)
                
        uncertainty_scores = self.process_batch(results)
        df = pd.DataFrame({'uncertainty_score': uncertainty_scores})
        df.to_csv(f"{self.experiment_name}/uncertainty_scores.csv", index=False)
    

        