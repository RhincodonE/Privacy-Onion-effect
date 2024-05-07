import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)  # Setup logging

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        logging.info(f"Directory already exists: {path}")

results_directory = '/users/home/ygu27/cifar/MIA/data/results'
ensure_directory(results_directory)  # Ensure the directory exists

csv_files = [os.path.join(results_directory, file) for file in os.listdir(results_directory) if file.endswith('.csv')]

correct_counts = {}
total_files = len(csv_files)

for file_path in csv_files:
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        sample_index = row['index']
        if row['True_label'] == row['Prediction']:
            if sample_index not in correct_counts:
                correct_counts[sample_index] = 0
            correct_counts[sample_index] += 1

# Initialize a dictionary to store the correct rate per index
correct_rate = {}
for index in correct_counts:
    correct_rate[index] = correct_counts[index] / total_files

# Optionally print the correct rate for each index
privacy_score_dir = '/users/home/ygu27/cifar/MIA/data/privacy_score'
ensure_directory(privacy_score_dir)
privacy_score_addr = os.path.join(privacy_score_dir,'scores.csv')

correct_rate_df = pd.DataFrame(list(correct_rate.items()), columns=['Index', 'CorrectRate'])

# Sort the DataFrame by 'Index'
correct_rate_df.sort_values(by='Index', inplace=True)

# Save the sorted DataFrame to CSV
correct_rate_df.to_csv(privacy_score_addr, index=False)
print(f"Correct rates saved to {privacy_score_addr}")
