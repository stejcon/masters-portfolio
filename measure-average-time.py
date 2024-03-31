import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate average time taken from a JSON file
def calculate_average_time(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        time_taken_values = [entry['time_taken'] for entry in data]
        average_time = sum(time_taken_values) / len(time_taken_values)
        return average_time * 1000  # Convert seconds to milliseconds

# Directory containing JSON files
directory = './results'

# Define categories and subcategories
models = ['resnet18', 'resnet34', 'resnet50']
datasets = ['cifar10', 'cifar100', 'qmnist', 'fashion']
variants = ['base', 'branched', 'stripped']

# Dictionary to store data
data = {model: {variant: {dataset: [] for dataset in datasets} for variant in variants} for model in models}

# Iterate through each JSON file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        model, dataset, variant = filename.split('-')
        with open(os.path.join(directory, filename), 'r') as f:
            time_taken = calculate_average_time(os.path.join(directory, filename))
            data[model][variant.split('.')[0]][dataset].append(time_taken)

# Plotting individual graphs for each model
for model in models:
    plt.figure(figsize=(12, 6))
    for idx, variant in enumerate(variants):
        avg_times = [np.mean(data[model][variant][dataset]) for dataset in datasets]
        plt.bar(np.arange(len(datasets)) + idx * 0.25, avg_times, width=0.25, label=variant)
    plt.title(f'{model.capitalize()}')
    plt.xlabel('Dataset')
    plt.ylabel('Average Time (ms)')
    plt.xticks(np.arange(len(datasets)) + 0.25, datasets)
    plt.legend()
    plt.tight_layout()
    plt.show()
