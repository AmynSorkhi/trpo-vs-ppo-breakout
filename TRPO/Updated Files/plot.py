import pandas as pd
import glob
import matplotlib.pyplot as plt
import json

path = "."

all_files = glob.glob(path + "/*.csv")

hyperparams_dict = {}
dataframes = []
min_epochs = 1000
for file in all_files:
    with open(file, 'r') as f:
        first_line = f.readline().strip()
        hyperparams = {k: v for param in first_line.split(',') if '=' in param for k, v in [param.split('=')]}
        hyperparams_dict[file] = hyperparams

    df = pd.read_csv(file)
    df = df[[col for col in df.columns if '=' not in col]]  # Keep only columns without '='
    df["Source_File"] = file
    df["Window Mean Reward"] = df["Mean Reward"].rolling(window=5).mean()  # Compute rolling mean
    dataframes.append(df)
    min_epochs = min(min_epochs, len(df))

with open("hyperparameters.json", "w") as json_file:
    json.dump(hyperparams_dict, json_file, indent=4)

for i in range(len(dataframes)):
    dataframes[i] = dataframes[i].iloc[:min_epochs]
combined_df = pd.concat(dataframes, ignore_index=True)

summary_stats = combined_df.describe()
print("Summary Statistics:\n", summary_stats)

metrics = ["Mean Reward", "Sample Efficiency", "Entropy", "Window Mean Reward"]
for metric in metrics:
    plt.figure(figsize=(12, 6))
    for file in all_files:
        exp_df = combined_df[combined_df["Source_File"] == file]
        plt.plot(exp_df["Epoch"], exp_df[metric], label=file)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"Comparison of {metric} Across Experiments")
    plt.legend()
    plt.grid()
    plt.show()

correlation_matrix = combined_df.drop(columns=["Source_File", "Epoch"]).corr()
print("\nCorrelation Matrix:\n", correlation_matrix)
