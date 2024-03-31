import json
import os
from collections import defaultdict
from tabulate import tabulate


# Function to process JSON file
def process_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Dictionary to store statistics for each exit number
    exit_stats = defaultdict(
        lambda: {
            "total_entropy": 0,
            "total_time": 0,
            "total_entries": 0,
            "correct_count": 0,
        }
    )

    # Calculate statistics
    for entry in data:
        exit_number = entry["exit_number"]
        exit_stats[exit_number]["total_entropy"] += entry["entropy"]
        exit_stats[exit_number]["total_time"] += entry["time_taken"]
        exit_stats[exit_number]["total_entries"] += 1
        if entry["correct"]:
            exit_stats[exit_number]["correct_count"] += 1

    total_entries = len(data)
    total_correct = sum(stats["correct_count"] for stats in exit_stats.values())
    total_time_taken = sum(entry["time_taken"] for entry in data)

    table = []
    headers = [
        "Exit Number",
        "Entries",
        "Percentage",
        "Avg Entropy",
        "Avg Time (ms)",
        "Accuracy (%)",
    ]
    # Generate statistics for each exit number
    for exit_number, stats in exit_stats.items():
        avg_entropy = stats["total_entropy"] / stats["total_entries"]
        avg_time_taken = stats["total_time"] / stats["total_entries"]
        accuracy = stats["correct_count"] / stats["total_entries"] * 100
        exit_percentage = stats["total_entries"] / total_entries * 100

        table.append(
            [
                exit_number,
                stats["total_entries"],
                f"{exit_percentage:.2f}%",
                avg_entropy,
                avg_time_taken * 1000,
                f"{accuracy:.2f}%",
            ]
        )

    overall_avg_time_taken = total_time_taken / total_entries * 1000
    overall_accuracy = total_correct / total_entries * 100

    print(f"Statistics for file: {file_path}")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Overall Average Time Taken: {overall_avg_time_taken:.4f} milliseconds")
    print()


# Get all JSON files in current directory
json_files = [file for file in os.listdir() if file.endswith(".json")]
json_files.sort()

# Process each JSON file
for file in json_files:
    process_json(file)
