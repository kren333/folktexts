import argparse
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import pdb

if __name__ == "__main__":
    # folders = ["acsincome/openai/gpt-4o-mini_bench-3466366301", "acspubcov/openai/gpt-4o-mini_bench-3064965676", "acsunemployment/openai/gpt-4o-mini_bench-1236485182",
    #            "diabetes_readmission/openai/gpt-4o-mini_bench-2624672895", "ipums/openai/gpt-4o-mini_bench-2320147224", "sepsis/openai/gpt-4o-mini_bench-2353442754"]
    d = 'zero_shot_results'

    datapoints = []

    for folder in os.listdir(d):

        folder_of_interest = f"{d}/{folder}"
        
        # read in risk scores df, metrics json
        for f in os.listdir(folder_of_interest):
            if f[-4:] == ".csv":
                risk_score_df = pd.read_csv(f"{folder_of_interest}/{f}")
            if f[-5:] == ".json":
                with open(f"{folder_of_interest}/{f}", "r") as fi:
                    metrics_json = json.load(fi)
        avg_risk = np.mean(np.abs(risk_score_df["risk_score"] - np.round(risk_score_df["risk_score"])))
        std_risk = np.std(risk_score_df["risk_score"])
        label_imbalance = max(np.mean(risk_score_df["label"]), 1-np.mean(risk_score_df["label"]))
        # avg_risk = np.mean(risk_score_df["risk_score"])
        auc = metrics_json["roc_auc"]      

        # add average risk, metric to datapoints
        datapoints.append([avg_risk, auc, std_risk, label_imbalance])

    # plot datapoints
    plt.scatter([(x[0]) for x in datapoints], [x[1] for x in datapoints])
    plt.xlabel("average(risk score - round(risk score))")
    plt.ylabel("auc score")
    plt.savefig("risk_score_correlation.png")
    plt.cla()

    plt.scatter([x[2] for x in datapoints], [x[1] for x in datapoints])
    plt.xlabel("sd(risk score)")
    plt.ylabel("auc score")
    plt.savefig("risk_std_correlation.png")
    plt.cla()

    plt.scatter([x[3] for x in datapoints], [x[1] for x in datapoints])
    plt.xlabel("frequency of majority class")
    plt.ylabel("auc score")
    plt.savefig("imbalance_correlation.png")
    plt.cla()
