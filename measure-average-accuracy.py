import json
import os
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate accuracy from a JSON file
def calculate_accuracy(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
        correct_count = sum(entry["correct"] for entry in data)
        total_count = len(data)
        accuracy = (correct_count / total_count) * 100
        return accuracy


# Directory containing JSON files
directory = "./results"

# Define categories and subcategories
models = ["resnet18", "resnet34", "resnet50"]
datasets = ["cifar10", "cifar100", "qmnist", "fashion"]
variants = ["base", "branched", "stripped"]

# Dictionary to store data
data = {
    model: {variant: {dataset: [] for dataset in datasets} for variant in variants}
    for model in models
}

# Iterate through each JSON file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        model, dataset, variant = filename.split("-")
        with open(os.path.join(directory, filename), "r") as f:
            accuracy = calculate_accuracy(os.path.join(directory, filename))
            data[model][variant.split(".")[0]][dataset].append(accuracy)

# Plotting individual graphs for each model
for model in models:
    plt.figure(figsize=(12, 6))
    for idx, variant in enumerate(variants):
        accuracies = [np.mean(data[model][variant][dataset]) for dataset in datasets]
        plt.bar(
            np.arange(len(datasets)) + idx * 0.25, accuracies, width=0.25, label=variant
        )
    plt.title(f"{model.capitalize()} - Accuracy")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.xticks(np.arange(len(datasets)) + 0.25, datasets)
    plt.legend()
    plt.tight_layout()
    plt.show()
