import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import logging

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

    else:
        logging.info(f"Directory already exists: {path}")

def custom_norm_fit(data):
    """
    Fit a normal distribution to the data and return the mean and standard deviation.

    :param data: List or array-like object containing the data to fit.
    :return: A tuple containing the mean and standard deviation of the fitted normal distribution.
    """
    # Check if data is a list of tensors and ensure it's not empty


    # Calculate the mean and standard deviation
    mu = np.mean(data)
    std = np.std(data, ddof=1)  # Using ddof=1 for sample standard deviation
    return mu, std

results_directory = '/users/home/ygu27/cifar/MIA/data/obs'

in_logit_gaps = []
out_logit_gaps = []

csv_files = [file for file in os.listdir(results_directory) if file.endswith('.csv')]

for file_path in csv_files:
    df = pd.read_csv(os.path.join(results_directory,file_path))

    # Step 3: Extract data and update the lists
    for i, row in df.iterrows():
        true_label = row['True_label']
        logit_gap = row['logit_gap']
        # Append logit gap to the appropriate list based on the true_label
        if  true_label == 'train_in':
            in_logit_gaps.append(logit_gap)
        elif true_label == 'train_out':
            out_logit_gaps.append(logit_gap)

out_mu,out_std = custom_norm_fit(out_logit_gaps)
in_mu,in_std = custom_norm_fit(in_logit_gaps)
stats_df_out = pd.DataFrame({
    'Parameter': ['mu', 'std'],
    'Value': [out_mu, out_std]
})
stats_df_in = pd.DataFrame({
    'Parameter': ['mu', 'std'],
    'Value': [in_mu, in_std]
})

ensure_directory('/users/home/ygu27/cifar/MIA/data/static')

csv_filename_out = '/users/home/ygu27/cifar/MIA/data/static/out_statistics.csv'
csv_filename_in = '/users/home/ygu27/cifar/MIA/data/static/in_statistics.csv'

stats_df_out.to_csv(csv_filename_out, index=False)
stats_df_in.to_csv(csv_filename_in, index=False)
