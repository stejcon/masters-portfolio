import json
import os
from collections import defaultdict


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

    print(f"Statistics for file: {file_path}")
    # Print statistics for each exit number
    for exit_number, stats in exit_stats.items():
        avg_entropy = stats["total_entropy"] / stats["total_entries"]
        avg_time_taken = stats["total_time"] / stats["total_entries"]
        accuracy = (
            stats["correct_count"] / stats["total_entries"] * 100
        )  # Accuracy in percentage
        exit_percentage = (
            stats["total_entries"] / total_entries * 100
        )  # Percentage of total entries
        print(f"Exit Number: {exit_number}")
        print(f"Number of Entries: {stats['total_entries']} ({exit_percentage:.2f}%)")
        print(f"Average Entropy: {avg_entropy}")
        print(f"Average Time Taken: {avg_time_taken * 1000:.4f} milliseconds")
        print(f"Accuracy: {accuracy:.2f}%")
        print()

    # Calculate overall statistics
    overall_avg_time_taken = (
        total_time_taken / total_entries * 1000
    )  # Convert to milliseconds

    overall_accuracy = total_correct / total_entries * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Overall Average Time Taken: {overall_avg_time_taken:.4f} milliseconds")
    print()


# Get all JSON files in current directory
json_files = [file for file in os.listdir() if file.endswith(".json")]

# Process each JSON file
for file in json_files:
    process_json(file)
